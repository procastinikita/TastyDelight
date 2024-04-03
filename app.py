from flask import Flask,request,render_template,session, redirect, url_for
from get_recipes import get_recs
app = Flask(__name__)



@app.route("/",methods=['GET'])
def home_page():
    return render_template('index1.html')

@app.route("/recommend",methods=['POST'])
def recommend():
    if request.method == 'POST':
      ingrid = request.form.get("ingredients")
      recipes = get_recs(ingrid)
      #dataframe=recipes.to_html(classes='table table-striped', index=False)
      recipe=recipes[['RecipeName','URL','image-url']]
      #table_html = recipes.to_html(classes='table table-bordered', index=False)
      recipe_list = recipe.to_dict(orient='records')
      return render_template('index1.html',recipe_list=recipe_list)
    else:
        return render_template('index1.html')

if __name__ == "__main__":
  app.run(debug=True)

