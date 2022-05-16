from flask import Flask,render_template,request
import Get_data as gd
import pridict as pdata
app=Flask(__name__)
@app.route("/")
def hello():
        T,H,A,R,I,P=gd.get_value()
        PT=pdata.predict_weather()
        HT=pdata.predict_humidity()
        return render_template("home.html",T=T,H=H,R=R,P=P,I=I,A=A,PT=PT,HT=HT)
if __name__=='__main__':
    app.run(debug=True)