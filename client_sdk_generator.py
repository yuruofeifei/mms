import json
import os


class ClientSDKGenerator(object):
    @staticmethod
    def generate(openapi_endpoints, sdk_lanugage):
        if not os.path.exists('build'):
            os.makedirs('build')
        f = open('build/openapi.json', 'w')
        json.dump(openapi_endpoints, f, indent=4)
        f.flush()
        os.system('java -jar utils/swagger-codegen-cli-2.2.1.jar generate \
                   -i build/openapi.json \
                   -o build \
                   -l ' + sdk_lanugage)