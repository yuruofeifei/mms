import json
import os


class ClientSDKGenerator(object):
    """Client SDK Generator using Swagger codegen tool
    """

    @staticmethod
    def generate(openapi_endpoints, sdk_lanugage):
        """Generate client sdk by given OpenAPI specification and target language.

        Parameters
        ----------
        openapi_endpoints : dict
            OpenAPI format api definition
        sdk_lanugage : string
            Target language for client sdk
        """

        # Serialize OpenAPI definition to a file
        if not os.path.exists('build'):
            os.makedirs('build')
        f = open('build/openapi.json', 'w')
        json.dump(openapi_endpoints, f, indent=4)
        f.flush()

        # Use Swagger codegen tool to generate client sdk in target language
        os.system('java -jar utils/swagger-codegen-cli-2.2.1.jar generate \
                   -i build/openapi.json \
                   -o build \
                   -l ' + sdk_lanugage)