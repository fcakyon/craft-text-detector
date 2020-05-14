from torch import from_numpy, load, no_grad
from torch.autograd import Variable
from torch.backends.cudnn import benchmark as cudnn_benchmark
from torch.cuda import empty_cache as empty_cuda_cache
from torch.nn import DataParallel
