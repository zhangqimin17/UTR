# Gaussian Prior


def run_gaussian_mf(train_matrix, train, test, k,
                    method="svi", lr=0.05, n_steps=2000, mae_tol=0.03):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    from torch.distributions import constraints
    from torch.autograd import Variable

    import pyro
    import pyro.distributions as dist
    import pyro.optim as optim

    from pyro.infer import SVI, Trace_ELBO

    from pyro.infer.mcmc.api import MCMC
    from pyro.infer.mcmc import NUTS

    # from preprocess import fill_unrated, control_value
    from eval_metrics import mae

    # from sklearn.model_selection import train_test_split

    pyro.set_rng_seed(1)
    assert pyro.__version__.startswith('1.4.0')

    sigma_u = torch.tensor(1.0)
    sigma_v = torch.tensor(1.0)

    def matrix_factorization_normal(train_matrix, k=k):

        m = train_matrix.shape[0]
        n = train_matrix.shape[1]

        u_mean = Variable(torch.zeros([m, k]))
        u_sigma = Variable(torch.ones([m, k]) * sigma_u)

        v_mean = Variable(torch.zeros([n, k]))
        v_sigma = Variable(torch.ones([n, k]) * sigma_v)

        u = pyro.sample("u", dist.Normal(
            loc=u_mean, scale=u_sigma).to_event(2))
        v = pyro.sample("v", dist.Normal(
            loc=v_mean, scale=v_sigma).to_event(2))

        expectation = torch.mm(u, v.t())
        sigma = pyro.sample("sigma", dist.Uniform(0.1, 0.5, validate_args=False))
        is_observed = (~np.isnan(train_matrix))
        is_observed = torch.tensor(is_observed)
        train_matrix = torch.tensor(train_matrix)
        train_matrix[~is_observed] = 0  # ensure all values are valid

        with pyro.plate("player1", m, dim=-2):
            with pyro.plate("player2", n, dim=-3):
                with pyro.poutine.mask(mask=is_observed):
                    pyro.sample("obs", dist.Normal(expectation, sigma),
                                obs=train_matrix)


    def guide_svi(train_matrix, k=k):
        m = train_matrix.shape[0]
        n = train_matrix.shape[1]

        u_mean = pyro.param('u_mean', torch.zeros([m, k]))
        u_sigma = pyro.param('u_sigma', torch.ones(
            [m, k]) * sigma_u, constraint=constraints.positive)

        v_mean = pyro.param('v_mean', torch.zeros([n, k]))
        v_sigma = pyro.param('v_sigma', torch.ones(
            [n, k]) * sigma_v, constraint=constraints.positive)

        sigma_loc = pyro.param('sigma_loc', torch.tensor(0.3),
                               constraint=constraints.positive)
        sigma_scale = pyro.param('sigma_scale', torch.tensor(0.05),
                                 constraint=constraints.positive)

        pyro.sample("u", dist.Normal(u_mean, u_sigma).to_event(2))
        pyro.sample("v", dist.Normal(v_mean, v_sigma).to_event(2))
        pyro.sample("sigma", dist.Normal(sigma_loc, sigma_scale, validate_args=False))


    def train_via_opt_svi(model, guide):
        m = train_matrix.shape[0]
        n = train_matrix.shape[1]
        pyro.clear_param_store()
        svi = SVI(model, guide, optim.Adam({"lr": lr}), loss=Trace_ELBO())

        loss_list = []
        mae_list = []
        player1dict = dict(zip(list(train_matrix.index), [i for i in range(m)])) 
        player2dict = dict(zip(list(train_matrix.columns), [i for i in range(n)])) 
        for step in range(n_steps):
            loss = svi.step(train_matrix.values)
            pred = []
            for i, j in test[["player1id", "player2id"]].itertuples(index=False):
                if i not in list(train_matrix.index) and j not in list(train_matrix.columns):
                    r = 0.5
                elif i in list(train_matrix.index) and j not in list(train_matrix.columns):
                    r = train[train['player1id'] == i].agg({'match_win_prob': lambda x: x.mean(skipna=True)})['match_win_prob']
                elif i not in list(train_matrix.index) and j in list(train_matrix.columns):
                    r = train[train['player2id'] == j].agg({'match_win_prob': lambda x: x.mean(skipna=True)})['match_win_prob']
                else:
                    r = torch.dot(pyro.param("u_mean")[player1dict[i], :], pyro.param("v_mean")[player2dict[j], :])
                    r = r.item()
                    if r < 0:
                        r = 0
                    if r > 1:
                        r = 1
                pred.append(r)
            test_mae = mae(test['match_win_prob'], pred)
            if step > 1000 and test_mae - min(mae_list) > mae_tol:
                print('[stop at iter {}] loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae)
                )
                break
            loss_list.append(loss)
            mae_list.append(test_mae)
            if step % 250 == 0:
                print('[iter {}]  loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae))
        return(loss_list, mae_list)

    loss_list, mae_list = train_via_opt_svi(matrix_factorization_normal, guide_svi)

    return loss_list, mae_list
