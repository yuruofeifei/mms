from mxnet_model_service import MXNetBaseService
from utils.mxnet_utils import Image


class MXNetVisionService(MXNetBaseService):
	def _preprocess(self, data):
		img_arr = []
		for image in data:
			img_arr.append(Image.read(image))
		return img_arr

	def _postprocess(self, data):
		return data


