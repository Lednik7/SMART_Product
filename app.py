import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField

from input_adj_pipeline import create_featured_datasets
from get_predictions import *

from model_questions.MQ import load_questions_model
from recommendation_system import get_recommendations
from smart_trainer.check_goal_name import *

import pickle

#  load data
with open("data/full_books_without_year.pkl", "rb") as f:
    books = pickle.load(f)

Model_Questions = load_questions_model("model_questions/")

class UserForm(FlaskForm):
    goal_name = TextField('Формулировка цели*')
    goal_result = TextField('Есть ли у Вас образ результата по этой цели?')
    goal_type = TextField('К какому типу запроса относится Ваша цель?')
    goal_first_step = TextField('Каким может быть первый шаг для достижения данной цели? С чего бы Вы начали?')
    goal_domain = TextField('К какой тематической области относится Ваша цель?')
    goal_obstacle = TextField('Какие Вы видите преграды для достижения этой цели? Что может помешать?')
    goal_time = TextField('Сколько времени, как Вам кажется, может занять достижение Вами данной цели?')
    submit = SubmitField('Получить оценку')

class GoalForm(FlaskForm):
    goal_check = TextField('Цель')
    submit = SubmitField('Проверить')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pers_target_secretkey'


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UserForm()
    if form.validate_on_submit():
        session['goal_name'] = form.goal_name.data
        session['goal_result'] = form.goal_result.data
        session['goal_type'] = form.goal_type.data
        session['goal_first_step'] = form.goal_first_step.data
        session['goal_domain'] = form.goal_domain.data
        session['goal_obstacle'] = form.goal_obstacle.data
        session['goal_time'] = form.goal_time.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    if not session['goal_name']:
        return redirect(url_for('index'))
    else:
        topic_raw = session['goal_domain']
        input_ = {
        'goal_name':session['goal_name'],
        'goal_result':session['goal_result'],
        'goal_type':session['goal_type'],
        'goal_first_step':session['goal_first_step'],
        'goal_domain':session['goal_domain'],
        'goal_obstacle':session['goal_obstacle'],
        'goal_time':session['goal_time'],
    }
        features, vectors, input_df = create_featured_datasets(input_)
        SMART_pred, edu_car_pred, abstract_pred, topics_pred = predict(features, vectors, topic_raw, input_df)
        
        if topics_pred: pass
        else: topics_pred = ['Не удалось определить тематику цели']

        questions_prediction = Model_Questions.predict(input_["goal_name"])

        return render_template('prediction.html', smart=SMART_pred, edu_car=edu_car_pred, 
                                abstract=abstract_pred, topics=topics_pred, questions=questions_prediction)

@app.route('/recommendation')
def recommendation():
    list_of_books = get_recommendations(books, session['goal_name'])
    return render_template('recommendation.html', book=list_of_books)

@app.route('/smart_trainer', methods=['GET', 'POST'])
def smart_trainer_index():
    form = GoalForm()
    if form.validate_on_submit():
        session['goal_check'] = form.goal_check.data
        return redirect(url_for('smart_trainer_check'))
    return render_template('smart_trainer.html', form=form)

@app.route('/smart_trainer/check')
def smart_trainer_check():
    if not session['goal_check']:
        return redirect(url_for('smart_trainer.html'))
    else:
        checker = load_name_checker("smart_trainer/")
        check_goal = checker.is_correct_input_name(session['goal_check'])
        if check_goal == 0:
            check_goal = 'Цель составлена некорректно! Попробуйте ещё раз'
        else:
            check_goal = 'Цель сформулирована верно! Закрепим успех?'
        return render_template('smart_trainer_check.html', check_goal=check_goal)

@app.route('/useful_links')
def useful_links():

    return render_template('useful_links.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
