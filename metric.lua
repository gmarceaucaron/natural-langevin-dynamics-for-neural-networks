local class = require 'class'

QDMetric = class('QDMetric')

-- th train.lua -metric qdop -numReg 1e-14 -learningRate 0.01 -gamma 0.01 -n_hidden 100
function QDMetric:__init(inputSize, outputSize, opt)
   
   self.metricRegul = opt.numReg or 1e-7 -- numerical regularization for online
   self.metricGamma = opt.metricGamma or 0.5 -- update rate of the metric
   
   -- self.initMetric = true -- flag for first update
   self.Mii = torch.Tensor(outputSize, inputSize):fill(1)
   self.M0i = torch.Tensor(outputSize, inputSize):fill(0)
   self.M00 = torch.Tensor(outputSize):fill(1)

   if opt.cuda then
      require 'cutorch'
      require 'cunn'

      self.Mii = self.Mii:cuda()
      self.M0i = self.M0i:cuda()
      self.M00 = self.M00:cuda()
   end
   self.opt = opt
end


function QDMetric:sampleNoise()
   
   -- Invert metric
   --local A00 = torch.rsqrt(torch.clamp(self.M00,self.numReg,1e25)) --TODO: better regularization
   local A00 = torch.rsqrt(self.M00)
   local A0i_inv = torch.cmul(A00:view(-1,1):expandAs(self.M0i),self.M0i)
   local Aii = torch.rsqrt((self.Mii-torch.pow(A0i_inv,2)):clamp(1e-12,1e25))
   --local Aii = torch.rsqrt(self.Mii-torch.pow(A0i_inv,2))
   local A0i = torch.cmul(torch.cmul(torch.mul(A00,-1):view(-1,1):expandAs(Aii),Aii),A0i_inv)
   
   -- Sample noise
   local v_bias = torch.randn(self.M00:size())
   local v_weight = torch.randn(self.Mii:size())

   if self.opt.cuda then
      v_bias = v_bias:cuda()
      v_weight = v_weight:cuda()
   end
   
   weight_noise = Aii:cmul(v_weight)
   bias_noise = torch.sum(A0i:cmul(v_weight),2) + A00:cmul(v_bias)
   
   return weight_noise, bias_noise
   
end


function QDMetric:updateMetric(input, gradOutput)
   
   -- Tensor dimension
   -- gradOuput: n_ex x n_out
   -- input: n_ex x n_in

   local miniSize = input:size(1)
   local addBuffer = torch.Tensor(input:size(1)):fill(1)
   if self.opt.cuda then
      addBuffer = addBuffer:cuda()
   end
   
   -- Update preconditioner C
   local gradOutputSqT = torch.pow(gradOutput*miniSize,2):t()/miniSize
   -- if self.initMetric then
   --    -- if self.langevin then
   --    -- 	 self.Mii:fill(self.Mii:size(2)+1)
   --    -- 	 self.M0i:fill(0)
   --    -- 	 self.M00:fill(self.Mii:size(2)+1)
   --    -- else
   --    temp = gradOutputSqT * addBuffer      
   --    self.Mii:copy(gradOutputSqT * torch.pow(input,2))
   --    self.M0i:copy(gradOutputSqT * input)
   --    self.M00:copy(gradOutputSqT * addBuffer)
   --    --end
   --    self.initMetric = false
   -- else
   self.Mii:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,torch.pow(input,2)):add(self.metricGamma*self.metricRegul)
   --self.M0i:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,input):add(self.metricGamma*self.metricRegul)
   self.M0i:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,input)

   --- BUG

   --self.M0i:zero()
   ---


   self.M00:addmv(1.-self.metricGamma,self.metricGamma,gradOutputSqT,addBuffer):add(self.metricGamma*self.metricRegul)
   --end
   local pseudotime = 1./(self.metricGamma*self.metricGamma)+1.0
   self.metricGamma = 1./math.sqrt(pseudotime)
   
   -- min_eigen = math.min(torch.min(self.Mii+self.numReg),torch.min(self.M00+self.numReg))
   -- max_eigen = math.max(torch.max(self.Mii+self.numReg),torch.max(self.M00+self.numReg))
   -- print(min_eigen/max_eigen)
     
end


function QDMetric:convertGradient(gradWeight, gradBias)
   
   local numerator = torch.add(torch.cmul(gradWeight,self.M00:view(-1,1):expandAs(gradWeight)),
   				  -1.0, torch.cmul(self.M0i,gradBias:view(-1,1):expandAs(self.M0i)))
   --local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),
      ---1.0,torch.pow(self.M0i,2)):clamp(self.numReg,1e25)
   local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),
				    -1.0,torch.pow(self.M0i,2))
   -- Apply preconditioner
   local rieGradWeight = numerator:cdiv(denominator)
   local temp = torch.cmul(self.M0i,rieGradWeight):sum(2)
   --local rieGradBias = gradBias:add(-1.,temp):cdiv(self.M00:clamp(self.numReg,1e25))
   local rieGradBias = gradBias:add(-1.,temp):cdiv(self.M00)

   return rieGradWeight, rieGradBias
end


-- define the DMetric class
DMetric = class('DMetric')
function DMetric:__init(inputSize, outputSize, opt)
   
   self.metricRegul = opt.numReg or 1e-7
   self.metricGamma = opt.metricGamma or 0.5
   
   -- self.initMetric = true
   self.Mii = torch.Tensor(outputSize, inputSize):fill(1)
   self.M00 = torch.Tensor(outputSize):fill(1)

   -- self.Mii = torch.Tensor(outputSize, inputSize)
   -- self.M00 = torch.Tensor(outputSize)
   
   if opt.cuda then
      require 'cutorch'
      require 'cunn'

      self.Mii = self.Mii:cuda()
      self.M00 = self.M00:cuda()
   end
   self.opt = opt
end

function DMetric:sampleNoise()

   local v_bias = torch.randn(self.M00:size())
   local v_weight = torch.randn(self.Mii:size())   

   if self.opt.cuda then
      v_bias = v_bias:cuda()
      v_weight = v_weight:cuda()
   end

   -- local weight_noise = torch.cmul(torch.rsqrt(self.Mii+self.numReg),v_weight)
   -- local bias_noise = torch.cmul(torch.rsqrt(self.M00+self.numReg), v_bias)
   local weight_noise = torch.cmul(torch.rsqrt(self.Mii),v_weight)
   local bias_noise = torch.cmul(torch.rsqrt(self.M00), v_bias)
   
   return weight_noise, bias_noise
   
end

function DMetric:updateMetric(input, gradOutput)
   
   -- Tensor dimension
   -- gradOuput: n_ex x n_out
   -- input: n_ex x n_in

   local miniSize = input:size(1)
   local addBuffer = torch.Tensor(input:size(1)):fill(1)
   if self.opt.cuda then
      addBuffer = addBuffer:cuda()
   end
   
   -- Update preconditioner C
   local gradOutputSqT = torch.pow(gradOutput*miniSize,2):t()/miniSize
   -- if self.initMetric then
   --    -- if self.langevin then
   --    -- 	 self.Mii:fill(self.Mii:size(2)+1)
   --    -- 	 self.M00:fill(self.Mii:size(2)+1)
   --    -- else
   --    self.Mii:copy(gradOutputSqT * torch.pow(input,2))
   --    self.M00:copy(gradOutputSqT * addBuffer)
   --    --      end
   --    self.initMetric = false
   -- else
   self.Mii:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,torch.pow(input,2)):add(self.metricGamma*self.metricRegul)
   self.M00:addmv(1.-self.metricGamma,self.metricGamma,gradOutputSqT,addBuffer):add(self.metricGamma*self.metricRegul)
--end
   local pseudotime = 1./(self.metricGamma*self.metricGamma)+1.0
   self.metricGamma = 1./math.sqrt(pseudotime)
   
end


function DMetric:convertGradient(gradWeight, gradBias)

   -- Apply preconditioner
   -- local rieGradWeight = torch.cdiv(gradWeight, self.Mii+self.numReg) --TODO: better reg
   -- local rieGradBias = torch.cdiv(gradBias, self.M00+self.numReg)

   local rieGradWeight = torch.cdiv(gradWeight, self.Mii) --TODO: better reg
   local rieGradBias = torch.cdiv(gradBias, self.M00)
   
   return rieGradWeight, rieGradBias

end


RMSMetric = class('RMSMetric')
-- th train.lua -metric rmsprop -numReg 1e-12 -learningRate 0.1 -metricGamma 0.01 -n_hidden 100
-- th train.lua -metric rmsprop -numReg 1e-14 -learningRate 100 -metricGamma 0.001 -n_hidden 100 -langevin -priorSigma2 1 -batchSize 100
function RMSMetric:__init(inputSize, outputSize, opt)
   
   self.metricRegul = opt.numReg or 1e-7 -- numerical regularization for online
   self.metricGamma = opt.metricGamma or 0.5 -- update rate of the metric
   -- Initialize some data structures
   self.initMetric = true -- flag for first update
   self.Mii = torch.Tensor(outputSize, inputSize):fill(1)
   self.M00 = torch.Tensor(outputSize):fill(1)
   
   if opt.cuda then
      require 'cutorch'
      require 'cunn'
      
      self.Mii = self.Mii:cuda()
      self.M00 = self.M00:cuda()
   end
   self.opt = opt
end

function RMSMetric:sampleNoise()

   -- Sample noise
   local v_bias = torch.randn(self.M00:size())
   local v_weight = torch.randn(self.Mii:size())   

   if self.opt.cuda then
      v_bias = v_bias:cuda()
      v_weight = v_weight:cuda()
   end
   
   local weight_noise = torch.cmul(torch.rsqrt(torch.sqrt(self.Mii)), v_weight)
   local bias_noise = torch.cmul(torch.rsqrt(torch.sqrt(self.M00)), v_bias)
   -- local weight_noise = torch.cmul(torch.rsqrt(torch.sqrt(self.Mii)+self.numReg), v_weight)
   -- local bias_noise = torch.cmul(torch.rsqrt(torch.sqrt(self.M00)+self.numReg), v_bias)

   return weight_noise, bias_noise

end

function RMSMetric:updateMetric2(gradWeight, gradBias)
   
   -- Tensor dimension
   -- gradOuput: n_ex x n_out
   -- input: n_ex x n_in
   
   -- if self.initMetric then
   --    self.Mii:copy(torch.pow(gradWeight,2))
   --    self.M00:copy(torch.pow(gradBias,2))
   --    self.initMetric = false
   -- else
   self.Mii:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradWeight,gradWeight):add(self.metricGamma*self.metricRegul)
   self.M00:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradBias,gradBias):add(self.metricGamma*self.metricRegul)
--end
   local pseudotime = 1./(self.metricGamma*self.metricGamma)+1.0
   self.metricGamma = 1./math.sqrt(pseudotime)
end

function RMSMetric:updateMetric(input, gradOutput)
   
   -- Tensor dimension
   -- gradOuput: n_ex x n_out
   -- input: n_ex x n_in
   local gradWeight = gradOutput:t()*input
   local gradBias = torch.sum(gradOutput,1):t()
   
   -- if self.initMetric then
   --    self.Mii:copy(torch.pow(gradWeight,2))
   --    self.M00:copy(torch.pow(gradBias,2))
   --    self.initMetric = false
   -- else
   self.Mii:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradWeight,gradWeight):add(self.metricGamma*self.metricRegul)
   self.M00:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradBias,gradBias):add(self.metricGamma*self.metricRegul)
      --end
   local pseudotime = 1./(self.metricGamma*self.metricGamma)+1.0
   self.metricGamma = 1./math.sqrt(pseudotime)
end


function RMSMetric:convertGradient(gradWeight, gradBias)

   -- Apply preconditioner
   local rieGradWeight = torch.cinv(torch.sqrt(self.Mii)):cmul(gradWeight)
   local rieGradBias = torch.cinv(torch.sqrt(self.M00)):cmul(gradBias)

   -- local rieGradWeight = torch.cinv(torch.sqrt(self.Mii)+self.numReg):cmul(gradWeight)
   -- local rieGradBias = torch.cinv(torch.sqrt(self.M00)+self.numReg):cmul(gradBias)


   return rieGradWeight, rieGradBias

end

EuclMetric = class('EuclMetric')
function EuclMetric:__init(inputSize, outputSize, opt)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.opt = opt
end

function EuclMetric:sampleNoise()

   -- Sample noise

   local v_bias = torch.randn(self.outputSize)
   local v_weight = torch.randn(self.outputSize, self.inputSize)

   if self.opt.cuda then
      v_bias = v_bias:cuda()
      v_weight = v_weight:cuda()
   end
   
   return v_weight, v_bias
end

function EuclMetric:updateMetric(input, gradOutput)
end

function EuclMetric:convertGradient(gradWeight, gradBias)
   return gradWeight, gradBias
end



-- -- define the DMetric class
-- DMetric = class('DMetric')
-- -- th train.lua -metric dop -numReg 1e-12 -learningRate 0.001 -metricGamma 0.01 -n_hidden 100
-- -- th train.lua -metric dop -numReg 1e-11 -learningRate 0.0000002 -metricGamma 0.01 -n_hidden 100 -langevin -priorSigma2 1 -batchSize 100
-- function DMetric:__init(inputSize, outputSize, opt)
   
--    self.numReg = opt.numReg or 1e-7
--    self.metricGamma = opt.metricGamma or 0.01
   
--    self.initMetric = true
--    self.Mii = torch.Tensor(outputSize, inputSize)
--    self.M00 = torch.Tensor(outputSize)
--    self.langevin = opt.langevin or false
-- end

-- function DMetric:sampleNoise()
   
--    local v_weight = torch.randn(self.Mii:size())   
--    local weight_noise = torch.rsqrt(self.Mii+self.numReg):cmul(v_weight)

--    local v_bias = torch.randn(self.M00:size())
--    local bias_noise = torch.rsqrt(self.M00+self.numReg):cmul(v_bias)
   
--    return weight_noise, bias_noise
   
-- end

-- function DMetric:updateMetric(input, gradOutput)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    --local miniSize = input:size(1)
--    local addBuffer = torch.Tensor(input:size(1)):fill(1)
   
--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t()--/miniSize
--    if self.initMetric then
--       -- if self.langevin then
--       -- 	 self.Mii:fill(self.Mii:size(2)+1)
--       -- 	 self.M00:fill(self.Mii:size(2)+1)
--       -- else
-- 	 self.Mii:copy(gradOutputSqT * torch.pow(input,2))
-- 	 self.M00:copy(gradOutputSqT * addBuffer)
-- --      end
--       self.initMetric = false
--    else
--       self.Mii:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,torch.pow(input,2))
--       self.M00:addmv(1.-self.metricGamma,self.metricGamma,gradOutputSqT,addBuffer)		 
--    end
     
-- end


-- function DMetric:convertGradient(gradWeight, gradBias)

--    -- Apply preconditioner
--    local rieGradWeight = torch.cdiv(gradWeight, self.Mii+self.numReg) --TODO: better reg
--    local rieGradBias = torch.cdiv(gradBias, self.M00+self.numReg)

--    return rieGradWeight, rieGradBias

-- end


-- function RMSMetric:convertGradient(gradWeight, gradBias)

--    -- Apply preconditioner
--    local rieGradWeight = torch.rsqrt(self.Mii+self.numReg):cmul(gradWeight)
--    local rieGradBias = torch.rsqrt(self.M00+self.numReg):cmul(gradBias)

--    return rieGradWeight, rieGradBias

-- end

-- RMSMetric = class('RMSMetric')
-- -- th train.lua -metric rmsprop -numReg 1e-12 -learningRate 0.1 -metricGamma 0.01 -n_hidden 100
-- -- th train.lua -metric rmsprop -numReg 1e-14 -learningRate 100 -metricGamma 0.001 -n_hidden 100 -langevin -priorSigma2 1 -batchSize 100
-- function RMSMetric:__init(inputSize, outputSize, opt)
   
--    self.numReg = opt.numReg or 1e-7 -- numerical regularization for online
--    self.metricGamma = opt.metricGamma or 0.01 -- update rate of the metric
   
--    -- Initialize some data structures
--    self.initMetric = true -- flag for first update
--    self.Mii = torch.Tensor(outputSize, inputSize)
--    self.M00 = torch.Tensor(outputSize)
--    self.langevin = opt.langevin or false
-- end

-- function RMSMetric:sampleNoise()

--    -- Sample noise
--    local v_weight = torch.randn(self.Mii:size())   
--    local weight_noise = torch.rsqrt(self.Mii+self.numReg):cmul(v_weight)

--    local v_bias = torch.randn(self.M00:size())
--    local bias_noise = torch.rsqrt(self.M00+self.numReg):cmul(v_bias)

--    return weight_noise, bias_noise

-- end

-- function RMSMetric:updateMetric(input, gradOutput)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    --local miniSize = input:size(1)

--    local gradWeight = (gradOutput:t()*input)--/miniSize
--    local gradBias = torch.sum(gradOutput,1):t()--/miniSize

--    if self.initMetric then

--       -- if self.langevin then
--       -- 	 self.Mii:fill(self.Mii:size(2)+1)
--       -- 	 self.M00:fill(self.Mii:size(2)+1)
--       -- else
--       self.Mii:copy(torch.pow(gradWeight,2))
--       self.M00:copy(torch.pow(gradBias,2))
-- --      end
--       initMetric = false
--    else
--       self.Mii:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradWeight,gradWeight)
--       self.M00:mul(1.-self.metricGamma):addcmul(self.metricGamma,gradBias,gradBias)
--    end
     
-- end

-- function RMSMetric:convertGradient(gradWeight, gradBias)

--    -- Apply preconditioner
--    local rieGradWeight = torch.rsqrt(self.Mii+self.numReg):cmul(gradWeight)
--    local rieGradBias = torch.rsqrt(self.M00+self.numReg):cmul(gradBias)

--    return rieGradWeight, rieGradBias

-- end



-- function DMetric:updateMetricLangevin(input, gradOutput, weightPrior, biasPrior)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in
   
--    local miniSize = input:size(1)
--    local addBuffer = torch.Tensor(miniSize):fill(1)
   
--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t()/miniSize
--    if self.initMetric then
--       local stdv = 1./math.sqrt(self.Mii:size(2)+1)
--       self.Mii:fill(self.Mii:size(2)+1)
--       self.M00:fill(self.Mii:size(2)+1)
--       self.initMetric = false
--       -- print(self.Mii:size())
--       -- print(self.M00:size())
--    -- else
--    --    self.Mii = (1. - self.metricGamma) * self.Mii + self.metricGamma * (gradOutputSqT * torch.pow(input,2) + torch.pow(weightPrior,2))
--    --    self.M00 = (1. - self.metricGamma) * self.M00 + self.metricGamma * (gradOutputSqT * addBuffer + torch.pow(biasPrior, 2))
--    end
   
--    -- min_eigen = math.min(torch.min(self.Mii+self.numReg),torch.min(self.M00+self.numReg))
--    -- max_eigen = math.max(torch.max(self.Mii+self.numReg),torch.max(self.M00+self.numReg))
--    -- print(min_eigen/max_eigen)
     
-- end

-- function DMetric:updateMetricLangevin2(input, gradOutput, weightPrior, biasPrior)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in
   
--    local miniSize = input:size(1)
--    local addBuffer = torch.Tensor(miniSize):fill(1)
   
--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t()/miniSize
--    if self.initMetric then
--       self.Mii = gradOutputSqT * torch.pow(input,2) + torch.pow(weightPrior,2)
--       self.M00 = gradOutputSqT * addBuffer + torch.pow(biasPrior, 2)
--       self.initMetric = false
--    else
--       self.Mii = (1. - self.metricGamma) * self.Mii + self.metricGamma * (gradOutputSqT * torch.pow(input,2) + torch.pow(weightPrior,2))
--       self.M00 = (1. - self.metricGamma) * self.M00 + self.metricGamma * (gradOutputSqT * addBuffer + torch.pow(biasPrior, 2))
--    end

--    -- min_eigen = math.min(torch.min(self.Mii+self.numReg),torch.min(self.M00+self.numReg))
--    -- max_eigen = math.max(torch.max(self.Mii+self.numReg),torch.max(self.M00+self.numReg))
--    -- print(min_eigen/max_eigen)
     
-- end

-- function QDMetric:updateMetricLangevin(input, gradOutput, weightPrior, biasPrior)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    -- print("Weight")
--    -- print("Prior")
--    -- print(weightPrior:size())
--    -- print(biasPrior:size())
   
--    local miniSize = input:size(1)
--    local addBuffer = torch.Tensor(miniSize):fill(1)
   
--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t()/miniSize
--    if self.initMetric then
--       self.Mii = gradOutputSqT * torch.pow(input,2) + torch.pow(weightPrior,2)
--       self.M0i = gradOutputSqT * input + torch.cmul(weightPrior,biasPrior:view(-1,1):expandAs(weightPrior))
--       self.M00 = gradOutputSqT * addBuffer + torch.pow(biasPrior, 2)
--       self.initMetric = false
--    else
--       -- self.Mii:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,torch.pow(input,2))
--       -- self.M0i:addmm(1.-self.metricGamma,self.metricGamma,gradOutputSqT,input)
--       -- self.M00:addmv(1.-self.metricGamma,self.metricGamma,gradOutputSqT,addBuffer)		 
--       self.Mii = (1. - self.metricGamma) * self.Mii + self.metricGamma * (gradOutputSqT * torch.pow(input,2) + torch.pow(weightPrior,2))
--       self.M0i = (1. - self.metricGamma) * self.M0i + self.metricGamma * (gradOutputSqT * input + torch.cmul(weightPrior,biasPrior:view(-1,1):expandAs(weightPrior)))
--       self.M00 = (1. - self.metricGamma) * self.M00 + self.metricGamma * (gradOutputSqT * addBuffer + torch.pow(biasPrior, 2))

--    end

--    -- min_eigen = math.min(torch.min(self.Mii+self.numReg),torch.min(self.M00+self.numReg))
--    -- max_eigen = math.max(torch.max(self.Mii+self.numReg),torch.max(self.M00+self.numReg))
--    -- print(min_eigen/max_eigen)
     
-- end
