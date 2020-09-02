import gpytorch
import numpy as np
import torch
import tqdm
from gpytorch.models import ApproximateGP
from scipy import io
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

avg_r2 = []
use_encoder = False
f = 'gss'

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

class ODVGPModel(ApproximateGP):
    def __init__(self, mean_inducing_points, covar_inducing_points, use_encoder=False):
        variational_strategy = make_orthogonal_vs(self, mean_inducing_points, covar_inducing_points,use_encoder)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if use_encoder:
            self.encoder=Encoder(args['input_dim'],args['z_dim'],args['hidden_dim'])
        else:
            self.encoder=None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ODVGPModel_Input(gpytorch.Module):
    def __init__(self, mean_inducing_points, covar_inducing_points, use_encoder=False):
        super(ODVGPModel_Input, self).__init__()
        self.gp_layer = ODVGPModel(mean_inducing_points, covar_inducing_points, use_encoder)
        self.use_encoder=use_encoder

    def forward(self, x):
        if self.use_encoder:
            x = self.gp_layer.encoder(x)
        res = self.gp_layer(x)
        return res


class MyDataset(Dataset):
    def __init__(self, trainX, trainy, trainId, trainOid):
        self.trainX = trainX
        self.trainy = trainy
        self.trainId = trainId
        self.trainOid = trainOid

    def __getitem__(self, i):
        return self.trainX[i], self.trainy[i], self.trainId[i], self.trainOid[i]  # the last index is the observation index

    def __len__(self):
        return len(self.trainy)


for seed in range(1):
# for seed in range(10,11):
    file = f'{f}_{seed}'
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

    args = {
        'batch_size': 500,
        'test_batch_size': 500,
        'input_dim': trainX.shape[1],
        'z_dim': 10,
        'hidden_dim':16,
        'cuda': False,
        'n_induce_mean': 100,
        'n_induce_var': 100,
        'epoch': 100,
        'lr':.1,
    }

    kmModel = KMeans(n_clusters=args['n_induce_mean'])
    kmModel.fit(trainX.numpy())
    mean_inducing_points = torch.from_numpy(kmModel.cluster_centers_).type(torch.float)

    kmModel = KMeans(n_clusters=args['n_induce_var'])
    kmModel.fit(trainX.numpy())
    covar_inducing_points = torch.from_numpy(kmModel.cluster_centers_).type(torch.float)

    if args['cuda']:
        trainX, trainy, testX, testy = trainX.cuda(), trainy.cuda(), testX.cuda(), testy.cuda()

    train_ds = MyDataset(trainX, trainy, trainId, trainOid)
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    test_ds = MyDataset(testX, testy, testId, testOid)
    test_loader = DataLoader(test_ds, batch_size=args['test_batch_size'], shuffle=False)

    def make_orthogonal_vs(model, mean_inducing_points, covar_inducing_points,use_encoder=False):
        if use_encoder:
            mean_inducing_points = torch.randn([mean_inducing_points.shape[0], args['z_dim']])
            covar_inducing_points = torch.randn([covar_inducing_points.shape[0], args['z_dim']])

        if args['cuda']:
            mean_inducing_points, covar_inducing_points = mean_inducing_points.cuda(), covar_inducing_points.cuda()

        covar_variational_strategy = gpytorch.variational.VariationalStrategy(
            model, covar_inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(0)),
            learn_inducing_locations=True
        )

        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            covar_variational_strategy, mean_inducing_points,
            gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(0))
        )
        return variational_strategy

    def train_and_test_approximate_gp(model_cls):
        model = model_cls(mean_inducing_points, covar_inducing_points,use_encoder)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=trainy.numel())

        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=args['lr'])

        if args['cuda']:
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Training
        model.train()
        likelihood.train()
        epochs_iter = tqdm(range(args['epoch']), desc=f"Training {model_cls.__name__}")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            t = time.time()
            for x_batch, y_batch, xid, oid in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                epochs_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
            print(time.time() - t)

        # Testing
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch, xid, oid in test_loader:
                preds = model(x_batch)
                means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]
        error = torch.mean(torch.abs(means - testy.cpu()))
        r2 = r2_score(testy.cpu().numpy(), means)
        print(f"Test {model_cls.__name__} MAE: {error.item()}")
        print(f'r2 score {model_cls.__name__} R2: {r2}')
        avg_r2.append(r2)
        return model

    model = train_and_test_approximate_gp(ODVGPModel_Input)

avg_r2 = np.array(avg_r2).mean()
print(f'the average r2 {avg_r2}')