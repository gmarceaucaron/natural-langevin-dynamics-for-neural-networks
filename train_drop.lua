require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'
dofile 'metriclinear.lua'

if not opt then
   --print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Dropout training')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-n_hidden', 100, 'Number of hidden units on every layer')
   cmd:option('-learningRate', 1., 'learning rate at t=0')
   cmd:option('-optim', 'vanilla', 'optimizer: {vanilla, adam}')
   cmd:option('-dropout', false, 'flag for using dropout')
   cmd:option('-metric', 'none', 'metric type: {eucl,rmsprop,dop,qdop}')
   cmd:option('-metricGamma', 0.01, 'update rate of the metric')
   cmd:option('-numReg', 1e-7, 'numerical regularization')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
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

if opt.dropout then 
   exp_desc = 'dropout_nhidden_'..opt.n_hidden..'_lr_'..opt.learningRate..'_numReg_'..opt.numReg..'_metric_'..opt.metric..'_metricGamma_'..opt.metricGamma..'_batchSize_'..opt.batchSize
else
   exp_desc = 'sgd_nhidden_'..opt.n_hidden..'_lr_'..opt.learningRate..'_numReg_'..opt.numReg..'_metric_'..opt.metric..'_metricGamma_'..opt.metricGamma..'_batchSize_'..opt.batchSize
end
   if not opt.mute then
      print(exp_desc)
   end
opt.langevin = false

trainLogger = optim.Logger(paths.concat('results', exp_desc..'.log'))
trainLogger:setNames({'epoch','trainLoss','trainAccuracy','testLoss','testAccuracy'})

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

inputs = torch.Tensor(opt.batchSize,trainSet:size(2),trainSet:size(3))
targets = torch.Tensor(opt.batchSize)   
-------------------------------------

---------------------------
-- Definition of the model
---------------------------
model = nn.Sequential()
model:add(nn.Reshape(nInput))

if opt.metric == 'none' then
   model:add(nn.Linear(nInput,opt.n_hidden,opt))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.ReLU())
   model:add(nn.Linear(opt.n_hidden,opt.n_hidden,opt))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.ReLU())
   model:add(nn.Linear(opt.n_hidden,10,opt))
   print('No Metric')
else
   model:add(nn.MetricLinear(nInput,opt.n_hidden,opt))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.ReLU())
   model:add(nn.MetricLinear(opt.n_hidden,opt.n_hidden,opt))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.ReLU())
   model:add(nn.MetricLinear(opt.n_hidden,10,opt))
   print('Metric')
end

if not opt.mute then
   print(model)
end
-------------------------------
-- Definition of the criterion
-------------------------------
criterion = nn.CrossEntropyCriterion()

if opt.cuda then
   model:cuda()
   criterion = criterion:cuda()
   trainSet = trainSet:cuda()
   trainSetLabel = trainSetLabel:cuda()
   testSet = testSet:cuda()
   testSetLabel = testSetLabel:cuda()
   inputs = inputs:cuda()
   targets = targets:cuda()
end

-- Retrieve the pointers to the parameters and gradParameters from the model for latter use
parameters,gradParameters = model:getParameters()

-- Learning function
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
	 local loss = criterion:forward(outputs, targets)
	 local dfdo = criterion:backward(outputs, targets)
	 model:backward(inputs,dfdo)
	 -- if not opt.mute then
	 --    print(loss)
	 -- end
	 return loss, gradParameters
      end
      if opt.optim == 'adam' then
	 optim.adam(feval, parameters, opt)
      else
	 local fx,dfdx=feval(parameters)	 
	 parameters:add(-opt.learningRate*dfdx)
      end
   end

   if not opt.mute then
      print("tick" .. sys.clock()-tick1)
   end
end

prevLoss = 10e12
for i = 1,opt.maxEpoch do
   model:evaluate()
   
   local trainPred = model:forward(trainSet)
   local trainLoss = criterion:forward(trainPred, trainSetLabel) 
   trainConfusion:batchAdd(trainPred, trainSetLabel)
   if not opt.mute then
      print("EPOCH: " .. i)
      print(trainConfusion)
      print(" + Train loss " .. trainLoss)
   else
      trainConfusion:updateValids()
   end

   local testPred = model:forward(testSet)
   local testLoss = criterion:forward(testPred, testSetLabel)    
   testConfusion:batchAdd(testPred, testSetLabel)
   if not opt.mute then
      print(testConfusion)
      print(" + Test loss " .. testLoss)
   else
      testConfusion:updateValids()
   end

   trainLogger:add{i, trainLoss, trainConfusion.totalValid * 100, testLoss, testConfusion.totalValid * 100}

   trainConfusion:zero()
   testConfusion:zero()

   model:training()
   train(i)
end

npy4th = require 'npy4th'
npy4th.savenpy(paths.concat('networks',exp_desc..'.npy'), parameters)
