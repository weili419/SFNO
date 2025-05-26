import torch.utils.data as tud
import argparse
from Utils import *
from CAVE_Dataset import cave_dataset
# from imageio import imsave
import torchvision
from SSIM import *	
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/root/data1/SSF/Dataset/Cave/Test/', type=str, help='path of the testing data')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=12, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 12)

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

model = torch.jit.load("Checkpoint/BestModel_ts.pt")
model = model.eval()
model = model.cuda()

psnr_total = 0
sam_total = 0
ergas_total = 0
# ssim_total = 0
k = 0

for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(LR.cuda(),RGB.cuda())
        result = out
        result = result.clamp(min=0., max=1.)
        for i in [5,10,15,20,25,30]:
            torchvision.utils.save_image(result[0,i,:,:],'Result/'+file_list[j]+'_'+str(i+1)+'.png')

        result = result.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        HR = HR.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    psnr = cal_psnr(result, HR)
    print(psnr)
    psnr_total = psnr_total + psnr
    sam = compute_sam(result, HR)
    sam_total = sam_total + sam
    ergas = compute_ergas(HR,result)
    ergas_total = ergas_total + ergas
    # ssim_v = ssim(result,HR.cuda())
    # ssim_total = ssim_total + ssim_v
    k = k + 1


print(k)
print("Avg PSNR = %.4f" % (psnr_total/k))
print("Avg SAM = %.4f" % (sam_total/k))
print("Avg ERGAS = %.4f" % (ergas_total/k))
# print("Avg SSIM = %.4f" % (ssim_total/k))
