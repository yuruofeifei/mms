from flask import Flask, request, jsonify, send_file
from request_handler import RequestHandler


class FlaskRequestHandler(RequestHandler):
    """Flask HttpRequestHandler for handling requests.
    """
    def __init__(self, app_name):
        """
        Contructor for Flask request handler.
        
        Parameters
        ----------
        app_name : string 
            App name for handler.
        """
        self.app = Flask(app_name)

    def start_handler(self, host, port):
        """
        Start request handler.

        Parameters
        ----------
        host : string 
            Host to setup handler.
        port: int
            Port to setup handler.
        """
        self.app.run(host=host, port=port)

    def add_endpoint(self, api_name, endpoint, callback, methods):
        """
        Add an endpoint for Flask

        Parameters
        ----------
        endpoint : string 
            Endpoint for handler. 
        api_name: string
            Endpoint ID for handler.

        callback: function
            Callback function for endpoint.

        methods: List
            Http request methods [POST, GET].
        """

        # Flask need to be passed with a method list
        assert isinstance(methods, list) 
        self.app.add_url_rule(endpoint, api_name, callback, methods=methods)

    def get_query_string(self, field=None):
        """
        Get query string from a request.

        Parameters
        ----------
        field : string 
            Get field data from query string.

        Returns
        ----------
        Object: 
            Field data from query string.
        """
        if field is None:
            return request.args

        return request.args[field]

    def get_form_data(self, field=None):
        """
        Get form data from request.
        
        Parameters
        ----------
        field : string 
            Get field data from form data

        Returns
        ----------
        Object: 
            Field data from form data.
        """
        form = {k: v[0] for k, v in dict(request.form).iteritems()}
        if field is None:
            return form

        return form[field]

    def get_file_data(self, field=None):
        """
        Get file data from request.
        
        Parameters
        ----------
        field : string 
            Get field data from file data.

        Returns
        ----------
        Object: 
            Field data from file data.
        """
        files = {k: v[0] for k, v in dict(request.files).iteritems()}
        if field is None:
            return files

        return files[field]


    def jsonify(self, response):
        """
        Jsonify a response.
        
        Parameters
        ----------
        response : Response 
            response to be jsonified.

        Returns
        ----------
        Response: 
            Jsonified response.
        """
        return jsonify(response)

    def send_file(self, file, mimetype):
        """
        Send a file in Http response.
        
        Parameters
        ----------
        file : Buffer 
            File to be sent in the response.

        mimetype: string
            Mimetype (Image/jpeg).

        Returns
        ----------
        Response: 
            Response with file to be sent.
        """
        return send_file(file, mimetype)
