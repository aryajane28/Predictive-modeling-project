from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/customer_churn_raw.csv"; OUT=ROOT/"outputs"; MODELS=ROOT/"models"; REPORTS=ROOT/"reports"
for x in [OUT,MODELS,REPORTS]: x.mkdir(exist_ok=True)
df=pd.read_csv(DATA).drop_duplicates(); X=df.drop(columns=["customer_id","churn"]); y=df.churn
nums=["age","income","tenure_years","support_calls","monthly_spend","has_auto_pay"]; cats=["contract_type"]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cats)])
models={"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),"Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=250,max_depth=8,random_state=42,n_jobs=-1)}
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); fitted={}; rows=[]
for name,m in models.items():
 pipe=Pipeline([("pre",pre),("model",m)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv); score=pipe.predict_proba(Xv)[:,1]; fitted[name]=(pipe,pred,score); rows.append({"model":name,"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0),"roc_auc":roc_auc_score(yv,score)})
r=pd.DataFrame(rows).sort_values('roc_auc',ascending=False); r.to_csv(REPORTS/'model_metrics.csv',index=False); best=r.iloc[0].model; pipe,pred,score=fitted[best]; joblib.dump(pipe,MODELS/'best_model.joblib')
ConfusionMatrixDisplay(confusion_matrix(yv,pred)).plot(); plt.title('Confusion Matrix — '+best); plt.tight_layout(); plt.savefig(OUT/'confusion_matrix.png',dpi=160); plt.close()
plt.figure(figsize=(8,6));
for name,(pipe,pred,score) in fitted.items(): RocCurveDisplay.from_predictions(yv,score,name=name,ax=plt.gca())
plt.plot([0,1],[0,1],'--',label='Random baseline'); plt.title('ROC Curves — Model Comparison'); plt.tight_layout(); plt.savefig(OUT/'roc_curves.png',dpi=160); plt.close()
rf=fitted['Random Forest'][0]; imp=pd.Series(rf.named_steps['model'].feature_importances_,index=rf.named_steps['pre'].get_feature_names_out()).sort_values(ascending=False).head(12).sort_values(); imp.plot(kind='barh',figsize=(9,6)); plt.title('Random Forest — Top Feature Importances'); plt.tight_layout(); plt.savefig(OUT/'feature_importance.png',dpi=160); plt.close()
json.dump({'best_model':best,'test_size':.2,'random_state':42},open(REPORTS/'best_model.json','w'),indent=2); print(r.to_string(index=False))
