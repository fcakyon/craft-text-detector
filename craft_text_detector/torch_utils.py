from torch.autograd import Variable
from torch.nn import DataParallel
from torch.backends.cudnn import benchmark as cudnn_benchmark
from torch import load
from torch import from_numpy
from torch import no_grad
from torch.cuda import empty_cache as empty_cuda_cache
