from mxnet_model_service import MXNetBaseService
from utils.mxnet_utils import Image


class MXNetVisionService(MXNetBaseService):
	def _preprocess(self, data):
		img_list = []
		for image in data:
			img_arr = Image.read(image)
			img_arr = Image.transform_shape(img_arr)
			img_list.append(img_arr)
		return img_list

	def _postprocess(self, data):
		return data


