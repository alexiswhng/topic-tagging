import os
from datetime import date
from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
    
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(2000), nullable=False)
    pub_date = db.Column(db.DateTime, nullable=False)
    link = db.Column(db.String, nullable=False)
    source = db.Column(db.String(200), nullable=False)
    tag = db.Column(db.String(20))

    def __str__(self) -> str:
        return f"{self.source}: {self.title}"

@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    pagination = Article.query.order_by(Article.pub_date.desc()).paginate(page=page, per_page=10)
    return render_template('index.html', pagination=pagination)

#everytime website is refreshed, it runs the "init_db.py" file ***

# @app.route('/<tag>/')

if __name__ =="__main__":
    app.run(debug=True)