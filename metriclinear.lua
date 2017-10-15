require 'nn'
require 'mathx'
dofile 'metric.lua'

-- Priors. A prior is a function pr such that calling pr(theta) on a tensor theta
-- returns a pair ln pr(theta), d/dtheta (ln pr(theta))
function gaussianPrior(mean, variance)
   local function theprior(theta)
      local logPrior=(theta-mean):norm()
      logPrior=-logPrior*logPrior/(2*variance)-.5*math.log(2.*math.pi*variance)*theta:nElement()
      local gradlogPrior=-(theta-mean)/variance
      return logPrior,gradlogPrior
   end
   return theprior
end
function conjugateGaussianPrior(alpha, beta, mean)
   local mean=mean or 0
   local beta=beta or .5
   local alpha = alpha or .5
   local function theprior(theta)
      local S=(theta-mean):norm()
      S=S*S
      local n=theta:nElement()
      logPrior=math.lgamma(.5*n+alpha)-math.lgamma(alpha)-(.5*n+alpha)*math.log(beta+.5*S)+alpha*math.log(beta)-.5*n*math.log(2.*math.pi)
      gradlogPrior=((.5*n+alpha)/(beta+.5*S))*(mean-theta)
      return logPrior,gradlogPrior
   end
   return theprior
end

local MetricLinear, parent = torch.class('nn.MetricLinear', 'nn.Linear')

function MetricLinear:__init(inputSize, outputSize, opt)
   parent.__init(self,inputSize, outputSize)

   if opt.metric == "dop" then
      self.metric = DMetric(inputSize, outputSize, opt)
   elseif opt.metric == "qdop" then
      self.metric = QDMetric(inputSize, outputSize, opt)
   elseif opt.metric == "rmsprop" then
      self.metric = RMSMetric(inputSize, outputSize, opt)
   elseif opt.metric == "eucl" then
      self.metric = EuclMetric(inputSize, outputSize, opt)
   else
      print("Metric " .. opt.metric .. " not found!")
      do return end
   end

   self.datasetSize = opt.datasetSize or 50000   
   self.langevin = opt.langevin or false

   if self.langevin then 
      local prior = opt.prior or 'gaussian'
      if prior == 'gaussian' then
	 print('gaussian')
	 local priorMu = opt.priorMu or 0.0
	 local priorSigma2 = opt.priorSigma2 or 0.0
	 if opt.priorSigma2 < 1e-12  then
	    self.priorSigma2 =  1./inputSize
	 end
	 self.prior = gaussianPrior(priorMu, priorSigma2)
	 self.priorBias = gaussianPrior(0,1)
      elseif prior == 'conjGaussian' then
	 print('conjGaussian')
	 local priorMean = opt.priorMean or 0
	 local priotBeta = opt.priorBeta or .5
	 local priorAlpha = opt.priorAlpha or .5
	 self.prior = conjugateGaussianPrior(priorMean, priorBeta, priorAlpha)
	 self.priorBias = conjugateGaussianPrior(priorMean, priorBeta, priorAlpha)
      end
   end
   self.lr = opt.lr or 1
   self.n_update = 0   
end

function MetricLinear:reset()
   stdv = 1./math.sqrt(self.weight:size(2))
   self.weight:normal(0, stdv)
   self.bias:zero()
   return self
end

function MetricLinear:accGradParameters(input, gradOutput)

   -- Tensor dimension
   -- gradOuput: n_ex x n_out
   -- input: n_ex x n_in
   local gradWeight = gradOutput:t()*input
   local gradBias = torch.sum(gradOutput,1):t()
   
   if self.langevin then
      -- local weightPrior = torch.add(self.weight,-1.0*self.priorMu)/self.priorSigma2
      -- local biasPrior = torch.add(self.bias,-1.0*self.priorMu)/self.priorSigma2
      local _,weightGradPrior = self.prior(self.weight)
      local _,biasGradPrior = self.priorBias(self.bias)
      
      gradWeight:add(-1./self.datasetSize, weightGradPrior)
      gradBias:add(-1./self.datasetSize, biasGradPrior)
   end
   self.metric:updateMetric(input, gradOutput)
   --self.metric:updateMetric(gradWeight, gradBias)
   local rieGradWeight, rieGradBias = self.metric:convertGradient(gradWeight, gradBias)
   
   if self.langevin then
      
      local weight_noise, bias_noise = self.metric:sampleNoise()
      self.gradWeight:copy(rieGradWeight:mul(-1.*self.lr):add(math.sqrt(2.*self.lr/self.datasetSize),weight_noise))
      self.gradBias:copy(rieGradBias:mul(-1.*self.lr):add(math.sqrt(2.*self.lr/self.datasetSize),bias_noise))
      
      self.n_update = self.n_update + 1
      if self.n_update % 10000 == 0 then
      	 self.lr = self.lr / 2
      print("Changing learning rate to " .. self.lr)
      end
   else
      self.gradWeight:copy(rieGradWeight)
      self.gradBias:copy(rieGradBias)
   end
   
end



-- function MetricLinear:accGradParameters(input, gradOutput)

--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    --local miniSize = input:size(1)
   
--    local gradWeight = (gradOutput:t()*input)--/miniSize   
--    local gradBias = torch.sum(gradOutput,1):t()--/miniSize

--    if self.langevin then
--       local weightPrior = torch.add(self.weight,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)
--       local biasPrior = torch.add(self.bias,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)
      
--       gradWeight = gradWeight + weightPrior
--       gradBias =  gradBias + biasPrior
--    end
--    self.metric:updateMetric(input, gradOutput)
--    rieGradWeight, rieGradBias = self.metric:convertGradient(gradWeight, gradBias)

--    if self.langevin then

--       local weight_noise, bias_noise = self.metric:sampleNoise()

--       self.gradWeight:copy(self.dt*rieGradWeight - torch.sqrt(2*self.dt/self.datasetSize)*weight_noise)
--       self.gradBias:copy(self.dt*rieGradBias - torch.sqrt(2*self.dt/self.datasetSize)*bias_noise)
--    else
--       -- Update parameters
--       self.gradWeight:copy(self.dt*rieGradWeight)
--       self.gradBias:copy(self.dt*rieGradBias)
--    end

--    -- self.n_update = self.n_update + 1
--    -- if self.n_update % 500 == 0 then
--    --    self.dt = self.dt / 2
--    -- end
-- end


-- function MetricLinear:accGradParameters(input, gradOutput)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    local miniSize = input:size(1)
   
--    -- Compute the gradient of loss and incorporate the gradient of prior
--    local gradWeight = (gradOutput:t()*input)/miniSize   
--    local gradBias = torch.sum(gradOutput,1):t()/miniSize

--    if self.langevin then
--       gradWeight = gradWeight + torch.add(self.weight,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)
--       gradBias =  gradBias + torch.add(self.bias,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)
--    end
--    self.metric:updateMetric(input, gradOutput)

--    rieGradWeight, rieGradBias = self.metric:convertGradient(gradWeight, gradBias)

--    if self.langevin then
      
--       -- Sample preconditioned noise
--       local weight_noise, bias_noise = self.metric:sampleNoise()
      
--       -- Update parameters
--       self.gradWeight:copy(self.dt*rieGradWeight - torch.sqrt(2*self.dt/self.datasetSize)*weight_noise)
--       self.gradBias:copy(self.dt*rieGradBias - torch.sqrt(2*self.dt/self.datasetSize)*bias_noise)
--    else
--       -- Update parameters
--       self.gradWeight:copy(self.dt*rieGradWeight)
--       self.gradBias:copy(self.dt*rieGradBias)
--    end
-- end


-- --th train.lua -gradient qdlangevin -numReg 1e-8 -dt 0.0001 -gamma 0.01 -n_hidden 10 -priorSigma2 1
-- --th train.lua -gradient qdlangevin -numReg 1e-8 -dt 0.0001 -gamma 0.01 -n_hidden 10 -priorSigma2 10 works better
-- function QDLangevinLayer:accGradParameters(input, gradOutput)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    local miniSize = input:size(1)
   
--    -- Compute the gradient of loss and incorporate the gradient of prior
--    local gradWeight = (gradOutput:t()*input)
--       + miniSize*torch.add(self.weight,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)
   
--    local gradBias = torch.sum(gradOutput,1):t()
--       + miniSize*torch.add(self.bias,-1.0*self.priorMu)/(self.datasetSize*self.priorSigma2)

--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t() -- TODO: In-place ???
--    if self.initMetric then
--       self.Mii = gradOutputSqT * torch.pow(input,2)
--       self.M0i = gradOutputSqT * input
--       self.M00:mv(gradOutputSqT,self.addBuffer)
--       self.initMetric = false
--    else
--       self.Mii:addmm(1.-self.gamma,self.gamma,gradOutputSqT,torch.pow(input,2))
--       self.M0i:addmm(1.-self.gamma,self.gamma,gradOutputSqT,input)
--       self.M00:addmv(1.-self.gamma,self.gamma,gradOutputSqT,self.addBuffer)		 
--    end
--    local numerator = torch.add(torch.cmul(gradWeight,self.M00:view(-1,1):expandAs(gradWeight)),
--    				  -1.0, torch.cmul(self.M0i,gradBias:view(-1,1):expandAs(self.M0i)))
--    local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),
--    				    -1.0,torch.pow(self.M0i,2)):clamp(self.numReg,1e25)
   
--    -- Apply preconditioner
--    local preGradWeight = numerator:cdiv(denominator)
--    local temp = torch.cmul(self.M0i,self.gradWeight):sum(2)
--    local preGradBias = gradBias:add(-1.,temp):cdiv(self.M00:clamp(self.numReg,1e25))
   
--    -- Sample preconditioned noise
--    local weight_noise, bias_noise = self:sampleNoise3()

--    -- Update parameters
--    self.gradWeight:copy(self.dt*preGradWeight - torch.sqrt(2*self.dt/self.datasetSize)*weight_noise)
--    self.gradBias:copy(self.dt*preGradBias - torch.sqrt(2*self.dt/self.datasetSize)*bias_noise)
  
-- end

-- th train.lua -gradient qdlangevin -numReg 1e-8 -dt 0.0001 -gamma 0.01 -n_hidden 10 -priorSigma2 1
-- function QDLangevinLayer:accGradParameters(input, gradOutput)
--    -- print(self.numReg)
--    -- print(self.dt)
--    -- print(self.gamma)
   
--    -- Tensor dimension
--    -- gradOuput: n_ex x n_out
--    -- input: n_ex x n_in

--    local miniSize = input:size(1)
   
--    -- Compute the gradient of loss and incorporate the gradient of prior
--    local gradWeight = (gradOutput:t()*input)
--    local gradBias = torch.sum(gradOutput,1):t()
   
--    -- Update preconditioner C
--    local gradOutputSqT = torch.pow(gradOutput,2):t() -- TODO: In-place ???
--    if self.initMetric then
--       self.Mii = gradOutputSqT * torch.pow(input,2)
--       self.M0i = gradOutputSqT * input
--       self.M00:mv(gradOutputSqT,self.addBuffer)
--       self.initMetric = false
--    else
--       self.Mii:addmm(1.-self.gamma,self.gamma,gradOutputSqT,torch.pow(input,2))
--       self.M0i:addmm(1.-self.gamma,self.gamma,gradOutputSqT,input)
--       self.M00:addmv(1.-self.gamma,self.gamma,gradOutputSqT,self.addBuffer)		 
--    end
--    local numerator = torch.add(torch.cmul(gradWeight,self.M00:view(-1,1):expandAs(gradWeight)),
--    				  -1.0, torch.cmul(self.M0i,gradBias:view(-1,1):expandAs(self.M0i)))
--    local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),
--    				    -1.0,torch.pow(self.M0i,2)):clamp(self.numReg,1e25)
   
--    -- Apply preconditioner
--    -- Why do I sum: probably to keep the semantic of Linear...???
--    local preGradWeight = numerator:cdiv(denominator)
--    local temp = torch.cmul(self.M0i,self.gradWeight):sum(2)
--    local preGradBias = gradBias:add(-1.,temp):cdiv(self.M00:clamp(self.numReg,1e25))
   
--    self.gradWeight:copy(self.dt*preGradWeight)
--    self.gradBias:copy(self.dt*preGradBias)
  
-- end

-- local function sampleNoise(self)

--    A00 = torch.Tensor(self.bias:size())
--    A0i = torch.Tensor(self.weight:size())
--    Aii = torch.Tensor(self.weight:size())
   
--    -- Invert metric (verify the iteration order)
--    print(self.weight:size(2))
--    for j=1,self.weight:size(2) do
--       A00[j] = 1./torch.sqrt(self.M00[j])
--       for i= 1,self.weight:size(1) do
-- 	 A0i_inv = A00[j] * self.M0i[i][j]
-- 	 Aii[i][j] = 1./torch.sqrt(self.Mii[i][j] - A0i_inv*A0i_inv)
-- 	 A0i[i][j] = -A00[j] * Aii[i][j] * A0i_inv
--       end
--    end

--    -- void VIQDOPLayer::updateInvMetric(){
--    --    for(unsigned j = 0; j < A0i_.cols(); j++){
--    -- 	 A00_(j) = 1./sqrt(VIM00_(j));
--    -- 	 for(unsigned i = 0; i < A0i_.rows(); i++){
--    -- 	    double A0i_inv = A00_(j) * VIM0i_(i,j);
--    -- 	    Aii_(i,j) = 1./sqrt(VIMii_(i,j) - A0i_inv*A0i_inv);
--    -- 	    A0i_(i,j) = -A00_(j) * Aii_(i,j) * A0i_inv;
--    -- 						  }
--    -- 					       }
--    -- 				      }   
   
--    -- Sample noise
--    weight_noise = torch.Tensor(self.weight:size())
--    bias_noise = torch.Tensor(self.bias:size())
--    for j =1,self.weight:size(2) do
--       v = torch.randn(self.weight:size(1)+1,1)
--       bias_noise[j] = A00[j]/torch.sqrt(self.datasetSize) * v[1]
--       for i=1,self.weight:size(1) do
-- 	 weight_noise[i][j] = Aii[i][j]/torch.sqrt(self.datasetSize) * v[i+1]
-- 	 bias_noise[j] = bias_noise[j] + A0i[i][j]/sqrt(self.datasetSize) * v[i+1]
--       end
--    end

--    weight_noise = torch.mul(weight_noise,torch.sqrt(2.0*self.dt))
--    bias_noise = torch.mul(bias_noise,torch.sqrt(2.0*self.dt))
--    -- for(unsigned j = 0; j < W_.cols(); j++){
--    --   MyVector v(W_.rows()+1);
--    --   param_sampler_->sampleStdNormal(v);
--    --   B_(j) = Bmu_(j) + (A00_(j)/sqrt(n_training_))*v(0);
--    --   for(unsigned i = 0; i < W_.rows(); i++){
--    --     W_(i,j) = Wmu_(i,j) + (Aii_(i,j)/sqrt(n_training_)) * v(i+1);
--    --     B_(j) += (A0i_(i,j)/sqrt(n_training_)) * v(i+1);
--    -- }
--    --   }

--    return weight_noise, bias_noise
   
-- end

-- function QDLangevinLayer:sampleNoise2()

--    A00 = torch.Tensor(self.bias:size())
--    A0i = torch.Tensor(self.weight:size())
--    Aii = torch.Tensor(self.weight:size())
   
--    -- Invert metric (verify the iteration order)
--    for j=1,self.weight:size(1) do
--       A00[j] = 1./torch.sqrt(self.M00[j])
--       for i= 1,self.weight:size(2) do
-- 	 A0i_inv = A00[j] * self.M0i[j][i]
-- 	 Aii[j][i] = 1./torch.sqrt(self.Mii[j][i] - A0i_inv*A0i_inv)
-- 	 A0i[j][i] = -A00[j] * Aii[j][i] * A0i_inv
--       end
--    end
   
--    -- Sample noise
--    weight_noise = torch.Tensor(self.weight:size())
--    bias_noise = torch.Tensor(self.bias:size())
--    for j =1,self.weight:size(1) do
--       v = torch.randn(self.weight:size(2)+1,1)
--       bias_noise[j] = A00[j]/torch.sqrt(self.datasetSize) * v[1]
--       for i=1,self.weight:size(2) do
-- 	 weight_noise[j][i] = Aii[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
-- 	 bias_noise[j] = bias_noise[j] + A0i[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
--       end
--    end

--    return weight_noise, bias_noise
   
-- end

-- function QDLangevinLayer:sampleNoise2()

--    A00 = torch.Tensor(self.bias:size())
--    A0i = torch.Tensor(self.weight:size())
--    Aii = torch.Tensor(self.weight:size())
   
--    -- Invert metric (verify the iteration order)
--    for j=1,self.weight:size(1) do
--       A00[j] = 1./torch.sqrt(self.M00[j])
--       for i= 1,self.weight:size(2) do
-- 	 A0i_inv = A00[j] * self.M0i[j][i]
-- 	 Aii[j][i] = 1./torch.sqrt(self.Mii[j][i] - A0i_inv*A0i_inv)
-- 	 A0i[j][i] = -A00[j] * Aii[j][i] * A0i_inv
--       end
--    end
   
--    -- Sample noise
--    weight_noise = torch.Tensor(self.weight:size())
--    bias_noise = torch.Tensor(self.bias:size())
--    for j =1,self.weight:size(1) do
--       v = torch.randn(self.weight:size(2)+1,1)
--       bias_noise[j] = A00[j]/torch.sqrt(self.datasetSize) * v[1]
--       for i=1,self.weight:size(2) do
-- 	 weight_noise[j][i] = Aii[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
-- 	 bias_noise[j] = bias_noise[j] + A0i[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
--       end
--    end

--    return weight_noise, bias_noise
   
-- end

-- function QDLangevinLayer:sampleNoise3()
   
--    -- Invert metric
--    local A00 = torch.rsqrt(self.M00)
--    local A0i_inv = torch.cmul(A00:view(-1,1):expandAs(self.M0i),self.M0i)
--    local Aii = torch.rsqrt(self.Mii-torch.pow(A0i_inv,2))
--    local A0i = torch.cmul(torch.cmul(torch.mul(A00,-1):view(-1,1):expandAs(Aii),Aii),A0i_inv)
   
--    -- for j=1,self.weight:size(1) do
--    --    A00[j] = 1./torch.sqrt(self.M00[j])
--    --    for i= 1,self.weight:size(2) do
--    -- 	 A0i_inv = A00[j] * self.M0i[j][i]
--    -- 	 Aii[j][i] = 1./torch.sqrt(self.Mii[j][i] - A0i_inv*A0i_inv)
--    -- 	 A0i[j][i] = -A00[j] * Aii[j][i] * A0i_inv
--    --    end
--    -- end
   
--    -- Sample noise
--    v_bias = torch.randn(self.bias:size())
--    v_weight = torch.randn(self.weight:size())
   
--    weight_noise = Aii:mul(1./torch.sqrt(self.datasetSize)):cmul(v_weight)
--    bias_noise = torch.sum(A0i:mul(1./torch.sqrt(self.datasetSize)):cmul(v_weight),2)
--       + torch.mul(A00, 1./torch.sqrt(self.datasetSize)):cmul(v_bias)

--    -- weight_noise = torch.cmul(torch.mul(Aii, 1./torch.sqrt(self.datasetSize)),v_weight) 
--    -- bias_noise = torch.sum(torch.cmul(torch.mul(A0i, 1./torch.sqrt(self.datasetSize)),v_weight),2)
--    --    + torch.cmul(torch.mul(A00, 1./torch.sqrt(self.datasetSize)),v_bias)
   
   
--    -- weight_noise = torch.Tensor(self.weight:size())
--    -- bias_noise = torch.Tensor(self.bias:size())
--    -- for j =1,self.weight:size(1) do
--    --    v = torch.randn(self.weight:size(2)+1,1)
--    --    bias_noise[j] = A00[j]/torch.sqrt(self.datasetSize) * v[1]
--    --    for i=1,self.weight:size(2) do
--    -- 	 weight_noise[j][i] = Aii[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
--    -- 	 bias_noise[j] = bias_noise[j] + A0i[j][i]/torch.sqrt(self.datasetSize) * v[i+1]
--    --    end
--    -- end

--    return weight_noise, bias_noise
   
-- end


-- function QDLangevinLayer:accGradParameters(input, gradOutput)
--    local gradOutputSqT = torch.pow(gradOutput,2):t()
--    if self.initMetric then
--       self.Mii:mm(gradOutputSqT,torch.pow(input,2))
--       self.M0i:mm(gradOutputSqT,input)
--       self.M00:mv(gradOutputSqT,self.addBuffer)
--       self.initMetric = false
--    else
--       self.Mii:addmm(1.-self.gamma,self.gamma,gradOutputSqT,torch.pow(input,2))
--       self.M0i:addmm(1.-self.gamma,self.gamma,gradOutputSqT,input)
--       self.M00:addmv(1.-self.gamma,self.gamma,gradOutputSqT,self.addBuffer)		 
--    end

--    local gradWeight = (self.datasetSize / input:size(1)) * gradOutput:t()*input
--       + torch.add(self.weight,-1.0*self.priorMu)/self.priorSigma2

--    local gradBias = (self.datasetSize / input:size(1))
--       * gradOutput:t()*self.addBuffer
--       + torch.add(self.bias,-1.0*self.priorMu)/self.priorSigma2

--    -- quasi-diagonal metric
--    local numerator = torch.add(torch.cmul(gradWeight,self.M00:view(-1,1):expandAs(gradWeight)),
-- 				  -1.0, torch.cmul(self.M0i,gradBias:view(-1,1):expandAs(self.M0i)))
--    local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),
-- 				    -1.0,torch.pow(self.M0i,2)):clamp(self.numReg,1e25)
--    self.gradWeight:add(numerator:cdiv(denominator):div(inputs:size(1)))	  
--    local temp = torch.cmul(self.M0i,self.gradWeight):sum(2)
--    self.gradBias:add(gradBias:add(-1.,temp):cdiv(self.M00:clamp(self.numReg,1e25)):div(inputs:size(1)))
   
--    -- QD inverse
--    local secondTermWeight, secondTermBias = self:sampleNoise3()

--    -- local secondTermWeight = torch.mul(torch.Tensor(self.gradWeight:size()):normal(),torch.sqrt(2.0*self.dt))
--    -- secondTermWeight:cdiv(torch.sqrt(self.datasetSize*self.Mii/input:size(1)+self.numReg))
--    self.gradWeight:copy(torch.add(self.dt*gradWeight,secondTermWeight))
--    --self.gradWeight:copy(self.dt*gradWeight)
      
--    -- local secondTermBias = torch.mul(torch.Tensor(self.gradBias:size()):normal(),torch.sqrt(2.0*self.dt))
--    -- secondTermBias:cdiv(torch.sqrt(self.datasetSize*self.M00/input:size(1)+self.numReg))
--    self.gradBias:copy(torch.add(self.dt*gradBias,secondTermBias))   	  
   
--    --self.gradBias:copy(self.dt*gradBias)
-- end
