import torch.cuda
from anti_useragent import UserAgent
ua = UserAgent()


# Opera/9.80 (X11; Linux i686; U; ru) Presto/2.8.131 Version/11.11

print(ua.chrome)


print(torch.cuda.is_available())