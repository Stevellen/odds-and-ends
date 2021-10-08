import torch
import torch.nn.functional as F
import torchvision.utils as TU
import torchvision.transforms.functional as TF

from PIL import Image
from imagenet_map import CLASS_MAP


class MaskMaker:
    def __init__(self, model, img_path, input_resolution=(384,384), topk=3, tau=0.001, batch_size=16, mask_size=4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.topk = topk
        self.tau = tau
        self.batch_size = batch_size
        self.mask_size = mask_size

        # Getting our input image and storing initial outputs
        self.image = Image.open(img_path)
        image_tensor = TF.pil_to_tensor(self.image.resize(input_resolution)).float()
        self.image_tensor = TF.normalize(image_tensor, mean=image_tensor.mean([1,2]), std=image_tensor.std([1,2]))
        self.image_tensor.requires_grad = True
        self.original_output = torch.softmax(self.get_model_preds(), dim=1)
        self.masked_img = None

        if self.device == 'cuda': # empty cache to free up memory
            torch.cuda.empty_cache()

    def get_model_preds(self):
        return self.model(self.image_tensor.unsqueeze(0).to(self.device))
    
    def get_topk_preds(self):
        return torch.topk(self.original_output, self.topk, 1)

    @torch.no_grad()
    def make_mask(self):
        top_scores = self.get_topk_preds().values
        top_idx = self.get_topk_preds().indices
        masked_img = torch.zeros_like(self.image_tensor)
        for batch, index in self.__get_mask_batch():
            outputs = torch.softmax(self.model(batch.to(self.device)), dim=1)
            diffs = (top_scores - outputs[:, top_idx]).abs().squeeze(1).norm(dim=1)
            # TODO: find a better way to do this piece; this is slow
            for idx, (ii, jj) in enumerate(index):
                masked_img[:, ii, jj] = diffs[idx]

            if self.device == 'cuda':
                torch.cuda.empty_cache()
        self.masked_img = masked_img
        return masked_img

    def get_masked_image(self, scale=False, as_img=False):
        if self.masked_img is None:
            self.make_mask()
        if scale:
            img = self.__scale_pixel_values()
        else:
            img = self.masked_img
        return self.__image_as_pil(img) if as_img else img

    def make_gradient_map(self, as_img=True):
        gradients = []
        for ii in self.get_topk_preds().values[0]:
            alphadot = self.original_output[0][ii.long()]
            gradients.append(
                torch.autograd.grad(outputs=alphadot, inputs=self.image_tensor, retain_graph=True)[0].detach()
            )
        gradmap = self.__scale_pixel_values(img=torch.stack(gradients).sum(0))
        if as_img:
            gradmap =  self.__image_as_pil(gradmap.sum(0))
        return gradmap

    def __image_as_pil(self, img):
        return TF.to_pil_image(img)

    def __get_mask_batch(self):
        mask_batch = []
        indices = []
        for i in torch.arange(0, self.image_tensor.shape[1], self.mask_size):
            for j in torch.arange(0, self.image_tensor.shape[2], self.mask_size):
                mask_batch.append(TF.erase(self.image_tensor, i, j, self.mask_size, self.mask_size, 0.))
                indices.append((slice(i,i+self.mask_size), slice(j,j+self.mask_size)))
                if len(mask_batch) == self.batch_size:
                    yield torch.stack(mask_batch), indices
                    mask_batch, indices = [], []
    
    def __scale_pixel_values(self, img=None):
        msk_img = self.masked_img.clone() if img is None else img
        for i in range(self.masked_img.size(0)):
            msk_img[i, ...] = (msk_img[i, ...] - msk_img[i, ...].min()) / (msk_img[i, ...].max() - msk_img[i, ...].min())
        return msk_img
        