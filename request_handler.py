from abc import ABCMeta, abstractmethod


class RequestHandler(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def add_endpoint(self, endpoint, api_name, callback, method='POST'):
		pass

	@abstractmethod
	def get_query_string(self):
		pass

	@abstractmethod
	def get_request_body(self):
		pass

	@abstractmethod
	def start_handler(self):
		pass