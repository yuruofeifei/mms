from abc import ABCMeta, abstractmethod


class RequestHandler(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def add_endpoint(self, endpoint, api_name, callback, method='POST'):
		pass

	@abstractmethod
	def get_form_data(self):
		pass

	@abstractmethod
	def get_file_data(self):
		pass

	@abstractmethod
	def start_handler(self):
		pass