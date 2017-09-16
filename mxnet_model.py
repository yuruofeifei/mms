import mms 

from model import SingleNodeModel, MultiNodesModel

import mxnet as mx


class MXNetModel(SingleNodeModel):
	def _predict(self, data):
		return mx.__version__
