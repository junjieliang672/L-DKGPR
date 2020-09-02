import math
import warnings
from typing import Any

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from scipy import io
from sklearn.metrics import r2_score
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

avg_r2 = []
for seed in range(5):
    file = f'tadpole_{seed}'
    data = io.loadmat(f'../{file}')
    trainX = torch.from_numpy(data['trainX']).type(torch.float)
    trainy = torch.from_numpy(data['trainY'].reshape(-1)).type(torch.float)
    testX = torch.from_numpy(data['testX']).type(torch.float)
    testy = torch.from_numpy(data['testY'].reshape(-1)).type(torch.float)
    trainId_Ori = data['trainId'].reshape(-1)
    testId_Ori = data['testId'].reshape(-1)
    trainOid_Ori = data['trainOid'].reshape(-1)
    testOid_Ori = data['testOid'].reshape(-1)
    allX = torch.cat([trainX, testX], dim=0)
    ally = torch.cat([trainy, testy], dim=0)
    ids = set(list(np.concatenate([trainId_Ori, testId_Ori])))
    oids = set(list(np.concatenate([trainOid_Ori, testOid_Ori])))
    idMap = {}
    i = 0
    for x in ids:
        idMap[x] = i
        i += 1
    trainId = torch.FloatTensor([idMap[x] for x in trainId_Ori]).type(torch.long)
    testId = torch.FloatTensor([idMap[x] for x in testId_Ori]).type(torch.long)
    minOid = np.min(list(oids))
    trainOid = torch.FloatTensor(trainOid_Ori - minOid)
    testOid = torch.FloatTensor(testOid_Ori - minOid)
    allIid = torch.cat([trainId, testId])
    allOid = torch.cat([trainOid, testOid])
    allmean = ally.mean()
    trainy -= allmean
    testy -= allmean

    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self, trainX, trainy, trainId, trainOid):
            self.trainX = trainX
            self.trainy = trainy
            self.trainId = trainId
            self.trainOid = trainOid

        def __getitem__(self, i):
            return self.trainX[i], self.trainy[i], self.trainId[i], self.trainOid[
                i]  # the last index is the observation index

        def __len__(self):
            return len(self.trainy)


    args = {
        'batch_size': int(np.min([len(trainX), 128])),
        'test_batch_size': int(np.min([len(trainX), 128])),
        'input_dim': trainX.shape[1],
        'z_dim': 5,
        'hidden_dim': 16,
        'cuda': True,
        'n_induce': 50,
        'epoch': 100,
        'lr': .1,
        'test_freq':10
    }

    train_ds = MyDataset(trainX, trainy, trainId, trainOid)
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    test_ds = MyDataset(testX, testy, testId, testOid)
    test_loader = DataLoader(test_ds, batch_size=args['test_batch_size'], shuffle=False)


    class Encoder(nn.Module):
        def __init__(self, input_dim, z_dim, hidden_dim):
            super(Encoder, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.CELU(),
                nn.Dropout(.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.CELU(),
                nn.Dropout(.2),
                nn.Linear(hidden_dim, z_dim)
            )

        def forward(self, x):
            return self.net(x)


    class DeepGaussianLikelihood(Likelihood):
        """
        Implements the Gaussian likelihood with a fully-connected layer.
        """

        def __init__(self, num_features=None, mixing_weights=True, mixing_weights_prior=None, **kwargs):
            super().__init__()
            self.num_classes = 1
            if mixing_weights:
                self.num_features = num_features
                if num_features is None:
                    raise ValueError("num_features is required with mixing weights")
                self.register_parameter(
                    name="mixing_weights",
                    parameter=torch.nn.Parameter(torch.randn(self.num_classes, num_features).div_(num_features)),
                )
                if mixing_weights_prior is not None:
                    self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")
            else:
                self.num_features = self.num_classes
                self.mixing_weights = None
            self.noise_covar = HomoskedasticNoise(noise_prior=None, noise_constraint=None, batch_shape=torch.Size())

        def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
            return self.noise_covar(*params, shape=base_shape, **kwargs)

        def forward(self, function_samples, *params, **kwargs):
            num_data, num_features = function_samples.shape[-2:]

            # Catch legacy mode
            if num_data == self.num_features:
                warnings.warn(
                    "The input to DeepGaussianLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                    "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                    DeprecationWarning,
                )
                function_samples = function_samples.transpose(-1, -2)
                num_data, num_features = function_samples.shape[-2:]

            if num_features != self.num_features:
                raise RuntimeError("There should be %d features" % self.num_features)

            if self.mixing_weights is not None:
                mixed_fs = function_samples @ self.mixing_weights.t()  # num_classes x num_data
            else:
                mixed_fs = function_samples
            mixed_fs = mixed_fs.reshape(mixed_fs.size()[:-1])
            noise = self._shaped_noise_covar(mixed_fs.shape, *params, **kwargs).diag()
            res = base_distributions.Normal(loc=mixed_fs, scale=noise.sqrt())
            return res

        def __call__(self, function, *params, **kwargs):
            if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
                warnings.warn(
                    "The input to DeepGaussianLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                    "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                    DeprecationWarning,
                )
                function = MultitaskMultivariateNormal.from_batch_mvn(function)
            return super().__call__(function, *params, **kwargs)


    class GaussianProcessLayer(gpytorch.models.ApproximateGP):
        def __init__(self, num_dim, grid_bounds, grid_size=64):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
            )

            # Our base variational strategy is a GridInterpolationVariationalStrategy,
            # which places variational inducing points on a Grid
            # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self, grid_size=grid_size, grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ), num_tasks=num_dim,
            )
            super().__init__(variational_strategy)

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                    )
                )
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.grid_bounds = grid_bounds

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


    class DKLModel(gpytorch.Module):
        def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
            super(DKLModel, self).__init__()
            self.feature_extractor = feature_extractor
            self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=args['n_induce'])
            self.grid_bounds = grid_bounds
            self.num_dim = num_dim

        def forward(self, x):
            features = self.feature_extractor(x)
            features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
            # This next line makes it so that we learn a GP for each feature
            features = features.transpose(-1, -2).unsqueeze(-1)
            res = self.gp_layer(features)
            return res


    num_features = args['z_dim']
    # num_features = args['input_dim']
    feature_extractor = Encoder(args['input_dim'], num_features, args['hidden_dim'])
    model = DKLModel(feature_extractor, num_dim=num_features)
    likelihood = DeepGaussianLikelihood(num_features=num_features)

    if args['cuda']:
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iterations = args['epoch']

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.covar_module.parameters(), 'lr': args['lr'] * 0.01},
        {'params': model.gp_layer.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=args['lr'], momentum=.9, nesterov=True, weight_decay=0)

    scheduler = MultiStepLR(optimizer, milestones=[0.5 * args['epoch'], 0.75 * args['epoch']], gamma=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=trainX.shape[0])


    def train():
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        with gpytorch.settings.num_likelihood_samples(8):
            for x, y, xid, oid in train_loader:
                if args['cuda']:
                    x = x.to('cuda')
                    y = y.to('cuda')

                optimizer.zero_grad()
                output = model(x)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()


    def test(save=False):
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for x, y, xid, oid in test_loader:
                if args['cuda']:
                    x = x.to('cuda')
                    y = y.to('cuda')
                output = model(x)
                likelihood_output = likelihood(output)
                preds = likelihood_output.mean.mean(0)
                means = torch.cat([means, preds.reshape(-1).cpu()])
        means = means[1:]

        r2 = r2_score(testy.cpu().numpy(), means)
        if save:
            avg_r2.append(r2)
        print(f'test r2: {r2}')


    for i in range(args['epoch']):
        with gpytorch.settings.use_toeplitz(False):
            train()
            if i % args['test_freq'] == 0:
                test()
        scheduler.step()

    test(True)

avg_r2 = np.array(avg_r2).mean()
print(f'the average r2 {avg_r2}')