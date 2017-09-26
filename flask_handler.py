from flask import Flask, request, jsonify, send_file
from request_handler import RequestHandler


class FlaskRequestHandler(RequestHandler):
	def __init__(self, app_name='mms'):
		self.app = Flask(app_name)

	def add_endpoint(self, api_name, endpoint, callback, methods=['POST']):
		self.app.add_url_rule(endpoint, api_name, callback, methods=methods)

	def get_query_string(self, name=None):
		if name is None:
			return request.args
		return request.args[name]

	def get_form_data(self, name):
		form = {k: v[0] for k, v in dict(request.form).iteritems()}
		if name is None:
			return form
		return form[name]

	def get_file_data(self, name):
		files = {k: v[0] for k, v in dict(request.files).iteritems()}
		if name is None:
			return files
		return files[name]

	def start_handler(self):
		self.app.run()

	def jsonify(self, response):
		return jsonify(response)

	def send_file(self, file, mimetype):
		return send_file(file, mimetype)
