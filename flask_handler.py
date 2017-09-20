from flask import Flask, request
from request_handler import RequestHandler


class FlaskRequestHandler(RequestHandler):
	def __init__(self, app_name='mms'):
		self.app = Flask(app_name)

	def add_endpoint(self, api_name, endpoint, callback, method='POST'):
		self.app.add_url_rule('/' + endpoint, api_name, callback)

	def get_query_string(self, name=None):
		if name is None:
			return request.args
		return request.args[name]

	def get_request_body(self, name=None):
		if name is None:
			return request.form
		return request.form[name]

	def start_handler(self):
		self.app.run()
