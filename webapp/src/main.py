from flask import Flask

app = Flask(__name__)


# Entry point
if __name__ == '__main__':
  app.run(debug=True)