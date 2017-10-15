require 'optim'
require 'nn'
dofile 'metriclinear.lua'

if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Natural Langevin Dynamics')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-n_hidden', 100, 'Number of hidden units on every layer')
   cmd:option('-lr', 1., 'learning rate at t=0')
   cmd:option('-metric', 'none', 'metric type: |eucl|rmsprop|dop|qdop')
   cmd:option('-prior', 'gaussian', 'type of prior: gaussian|conjGaussian')
   cmd:option('-priorMean', 0.0, 'Prior mean')
   cmd:option('-priorSigma2', 0.0, 'Prior Gaussian variance')
   cmd:option('-priorAlpha', 0.5, 'Prior ConjugateGaussian alpha')
   cmd:option('-priorBeta', 0.5, 'Prior ConjugateGaussian beta')
   cmd:option('-metricGamma', 0.5, 'update rate of the metric')
   cmd:option('-meanGamma', 1.0, 'update rate of the posterior mean')
   cmd:option('-numReg', 1e-7, 'numerical regularization')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-thinning', 100, 'put a model in the ensemble every n updates')
   cmd:option('-burnin', 500, 'number of updates before beginning to put models in the ensemble')
   cmd:option('-maxEpoch', 1000, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-save', "result.log", 'filename for the training performances')
   cmd:option('-test', false, 'flag for using the test set')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-mute', false, 'flag for muting the outputs to the standard output')
   cmd:option('-cuda', false, 'flag for using the cuda technology')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.manualSeed(opt.seed)

exp_desc = 'nhidden_'..opt.n_hidden..'_lr_'..opt.lr..'_numReg_'..opt.numReg..'_metric_'..opt.metric..'_metricGamma_'..opt.metricGamma..'_batchSize_'..opt.batchSize..'_burning_'..opt.burnin..'_thinning_'..opt.thinning..'_meanGamma_'..opt.meanGamma..'_prior_'..opt.prior..'_priorMean_'..opt.priorMean
if opt.prior == 'gaussian' then
   exp_desc = exp_desc..'_priorSigma2_'..opt.priorSigma2
elseif opt.prior == 'conjGaussian' then
   exp_desc = exp_desc..'_priorAlpha_'..opt.priorAlpha..'_priorBeta_'..opt.priorBeta
end

if not opt.mute then
   print(exp_desc)
end
opt.langevin = true
meanGamma = opt.meanGamma

trainLogger = optim.Logger(paths.concat('results', exp_desc..'.log'))
trainLogger:setNames({'epoch','currTrainLoss','currTrainAccuracy','currValidLoss','currValidAccuracy','pmTrainLoss','pmTrainAccuracy','pmValidLoss','pmValidAccuracy','ensTrainLoss','ensTrainAccuracy','ensValidLoss','ensValidAccuracy'})

--------------------------------------
-- Loading and normalizing the dataset
local mnist = require 'mnist'
local mnistDataset = mnist.traindataset()
local nInput = mnistDataset.data:size(2) * mnistDataset.data:size(3)

-- classes
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
testConfusion = optim.ConfusionMatrix(classes)

local nTrain, nTest
if opt.test == false then
   nTrain = 50000
   nTest = 10000

   trainSet = torch.Tensor(nTrain,mnistDataset.data:size(2),mnistDataset.data:size(3))
   trainSet:copy(mnistDataset.data:narrow(1,1,nTrain):float():div(255.))
   trainSetLabel = torch.Tensor(nTrain)
   trainSetLabel:copy(mnistDataset.label:narrow(1,1,nTrain))
   trainSetLabel:add(1)
   
   testSet = torch.Tensor(nTest,mnistDataset.data:size(2),mnistDataset.data:size(3))
   testSet:copy(mnistDataset.data:narrow(1,nTrain+1,nTest):float():div(255.))
   testSetLabel = torch.Tensor(nTest)
   testSetLabel:copy(mnistDataset.label:narrow(1,nTrain+1,nTest))
   testSetLabel:add(1)
else
   local mnistTest = mnist.testdataset()

   nTrain = mnistDataset.data:size(1)
   nTest = mnistTest.data:size(1)

   trainSet = torch.Tensor(nTrain,mnistDataset.data:size(2),mnistDataset.data:size(3))
   trainSet:copy(mnistDataset.data:float():div(255.))
   trainSetLabel = torch.Tensor(nTrain)
   trainSetLabel:copy(mnistDataset.label:narrow(1,1,nTrain))
   trainSetLabel:add(1)
   
   testSet = torch.Tensor(nTest,mnistTest.data:size(2),mnistTest.data:size(3))
   testSet:copy(mnistTest.data:float():div(255.))
   testSetLabel = torch.Tensor(nTest)
   testSetLabel:copy(mnistTest.label)
   testSetLabel:add(1)
end
opt.datasetSize = nTrain

print(opt.datasetSize)

inputs = torch.Tensor(opt.batchSize,trainSet:size(2),trainSet:size(3))
targets = torch.Tensor(opt.batchSize)
-------------------------------------

---------------------------
-- Definition of the model
---------------------------
model = nn.Sequential()
model:add(nn.Reshape(nInput))
model:add(nn.MetricLinear(nInput,opt.n_hidden,opt))
model:add(nn.ReLU())
model:add(nn.MetricLinear(opt.n_hidden,opt.n_hidden,opt))
model:add(nn.ReLU())
model:add(nn.MetricLinear(opt.n_hidden,10,opt))

if not opt.mute then
   print(model)
end
softmax = nn.SoftMax() -- required for averaging the output probabilities
-------------------------------
-- Definition of the criterion
-------------------------------
criterion = nn.CrossEntropyCriterion()
criterion_test = nn.ClassNLLCriterion()

if opt.cuda then
   require 'cutorch'
   require 'cunn'

   model:cuda()
   criterion = criterion:cuda()
   trainSet = trainSet:cuda()
   trainSetLabel = trainSetLabel:cuda()
   testSet = testSet:cuda()
   testSetLabel = testSetLabel:cuda()
   inputs = inputs:cuda()
   targets = targets:cuda()
   criterion_test = criterion_test:cuda()
   softmax = softmax:cuda()
end

-- Retrieve the pointers to the parameters and gradParameters from the model for latter use
parameters,gradParameters = model:getParameters()
tempParams = parameters:clone()
meanParameters = parameters:clone()

-- Ensemble parameters
ens = {}
n_thin = 0
n_burnin = 0

function train(epoch)

   local tick1 = sys.clock()
   
   -- It may help to shuffle the examples
   shuffle = torch.randperm(trainSet:size(1))
   
   for t = 1,trainSet:size(1),opt.batchSize do

      if not opt.mute then
	 xlua.progress(t,trainSet:size(1))
      end
	 
      -- Define the minibatch
      for i = 1,opt.batchSize do
	 inputs[i]:copy(trainSet[shuffle[t+i-1]])
	 targets[i] = trainSetLabel[shuffle[t+i-1]]
      end

      -- Definition of the evaluation function (closure)
      local feval = function(x)
	 
	 if parameters~=x then
	    parameters:copy(x)
	 end
	 
	 gradParameters:zero()
	 
	 local outputs = model:forward(inputs) 
	 -- local trainPred = softmax:forward(model:forward(trainSet))
	 -- print(outputs)

	 local loss = criterion:forward(outputs, targets)
	 local dfdo = criterion:backward(outputs, targets)
	 model:backward(inputs,dfdo)

	 return loss, gradParameters
      end

      local fx,dfdx=feval(parameters)	 
      parameters:add(dfdx)

      -- Update the posterior mean
      meanParameters:mul(1.-meanGamma):add(meanGamma, parameters)
      local pseudotime = 1./(meanGamma*meanGamma)+1
      meanGamma = 1./math.sqrt(pseudotime)

      -- Procedure for adding the model to the ensemble
      if n_burnin == opt.burnin then
	 if n_thin == opt.thinning then
	    table.insert(ens, parameters:clone())
	    n_thin = 0
	 else
	    n_thin = n_thin + 1
	 end
      else
	 n_burnin = n_burnin + 1
      end
   end

   if not opt.mute then
      print("tick" .. sys.clock()-tick1)
   end
end

prevLoss = 10e12
for i = 1,opt.maxEpoch do
   model:evaluate()

   -- Save the current parameters
   tempParams:copy(parameters)

   --- Evaluation of the current model
   currTrainPred = model:forward(trainSet):clone()
   currTrainLoss = criterion:forward(currTrainPred, trainSetLabel) 
   
   currTestPred = model:forward(testSet):clone()
   currTestLoss = criterion:forward(currTestPred, testSetLabel)

   trainConfusion:batchAdd(currTrainPred, trainSetLabel)
   if not opt.mute then
      print("EPOCH: " .. i)
      print(trainConfusion)
      print(" + Current Train loss " .. currTrainLoss)
   else
      trainConfusion:updateValids()
   end
   local currTrainAcc = trainConfusion.totalValid * 100
   
   testConfusion:batchAdd(currTestPred, testSetLabel)
   if not opt.mute then
      print(testConfusion)
      print(" + Test loss " .. currTestLoss)
   else
      testConfusion:updateValids()
   end
   local currTestAcc = testConfusion.totalValid * 100
   trainConfusion:zero()
   testConfusion:zero()
   ---

   --- Evaluation of the posterior mean model
   parameters:copy(meanParameters)
   local pmTrainPred = model:forward(trainSet):clone()
   local pmTrainLoss = criterion:forward(pmTrainPred, trainSetLabel) 

   local pmTestPred = model:forward(testSet):clone()
   local pmTestLoss = criterion:forward(pmTestPred, testSetLabel)    

   trainConfusion:batchAdd(pmTrainPred, trainSetLabel)
   if not opt.mute then
      print("EPOCH: " .. i)
      print(trainConfusion)
      print(" + Current Train loss " .. pmTrainLoss)
   else
      trainConfusion:updateValids()
   end
   local pmTrainAcc = trainConfusion.totalValid * 100
   
   testConfusion:batchAdd(pmTestPred, testSetLabel)
   if not opt.mute then
      print(testConfusion)
      print(" + Test loss " .. pmTestLoss)
   else
      testConfusion:updateValids()
   end
   local pmTestAcc = testConfusion.totalValid * 100
   trainConfusion:zero()
   testConfusion:zero()
   ---

   --- Evaluation of the ensemble
   local trainPredAvg = nil
   local trainLossAvg = nil

   local testPredAvg = nil
   local testLossAvg = nil

   if #ens > 0 then
      timer = torch.Timer()
      for j = 1, #ens do

	 parameters:copy(ens[j])

	 local trainPred = softmax:forward(model:forward(trainSet))
	 
	 if trainPredAvg == nil then
	    trainPredAvg = trainPred:clone()
	 else
	    trainPredAvg:add(trainPred)
	 end

	 local testPred = softmax:forward(model:forward(testSet))
	 if testPredAvg == nil then
	    testPredAvg = testPred:clone()
	 else
	    testPredAvg:add(testPred)
	 end
      end

      trainPredAvg:div(#ens)
      trainLossAvg = criterion_test:forward(torch.log(trainPredAvg), trainSetLabel)

      testPredAvg:div(#ens)
      testLossAvg = criterion_test:forward(torch.log(testPredAvg), testSetLabel)

      print('Evaluating ' .. #ens .. ' ensemble in ' .. timer:time().real .. ' seconds')

   else
      -- No element in the ensemble, evaluate current point
      parameters:copy(tempParams)
      trainPredAvg = model:forward(trainSet):clone()
      trainLossAvg = criterion:forward(trainPredAvg, trainSetLabel) 

      testPredAvg = model:forward(testSet):clone()
      testLossAvg = criterion:forward(testPredAvg, testSetLabel)
   end

   trainConfusion:batchAdd(trainPredAvg, trainSetLabel)
   if not opt.mute then
      print("EPOCH: " .. i)
      print(trainConfusion)
      print(" + Train loss " .. trainLossAvg)
   else
      trainConfusion:updateValids()
   end

   testConfusion:batchAdd(testPredAvg, testSetLabel)
   if not opt.mute then
      print(testConfusion)
      print(" + Test loss " .. testLossAvg)
   else
      testConfusion:updateValids()
   end
   
   trainLogger:add{i, currTrainLoss, currTrainAcc, currTestLoss, currTestAcc, pmTrainLoss, pmTrainAcc, pmTestLoss, pmTestAcc, trainLossAvg, trainConfusion.totalValid * 100, testLossAvg, testConfusion.totalValid * 100}

   trainConfusion:zero()
   testConfusion:zero()
   
   parameters:copy(tempParams)

   model:training()
   train(i)
end

npy4th = require 'npy4th'
npy4th.savenpy(paths.concat('networks',exp_desc..'.npy'), parameters)
