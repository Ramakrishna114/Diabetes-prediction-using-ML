#importing libraries
from tkinter import*
import tkinter as tk
from tkinter import ttk
from typing_extensions import dataclass_transform
from PIL import Image,ImageTk
import os
import pickle

# # *********** machine learning code ****************-------
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pylab as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# df3=pd.read_csv("diabetes.csv")
# df1=pd.read_csv("diabetes1.csv")
# df2=pd.read_csv("diabetes2.csv")

# df=pd.concat([df1,df2,df3],ignore_index=True)
# diabetes_df_copy = df.copy(deep = True)
# diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
# diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
# diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
# diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
# diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)

# X =df.drop(columns = 'Outcome', axis=1)
# Y = df['Outcome']

# # STandardsing the data
# scaler = StandardScaler()
# scaler.fit(X)
# standardized_data = scaler.transform(X)
# # print(standardized_data)

# X = standardized_data
# Y = df['Outcome']

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=2)

# DTC=DecisionTreeClassifier()
# model=DTC.fit(X_train,Y_train)

# from sklearn.metrics import accuracy_score
# X_train_prediction = DTC.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# #print('Accuracy score on DTC  training data : ', training_data_accuracy)
# X_test_prediction = DTC.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# #print('Accuracy score on DTC the test data : ', test_data_accuracy)




# -----++++++++++++++//////// ml end  ////////////////--------------+++++++++++   
# import pickle    

# Model = pickle.dumps(model)


# DF = pd.DataFrame()
# def action():
#     import pandas as pd
#     global DB
        
#     preg_var=tk.StringVar() 
#     glu_var=tk.StringVar() 
#     bp_var=tk.StringVar()
#     stn_var=tk.StringVar()
#     insulin_var=tk.StringVar()
#     bmi_var=tk.StringVar()
#     dpf_var=tk.StringVar()
#     age_var=tk.StringVar()        
#     DF = pd.DataFrame(columns=['Preg','glu','bp','skin','insulin','mass','pedi','age'])
#     PREG=preg_var.get()
#     DF.loc[0,'Preg']=PREG
#     GLU=glu_var.get()
#     DF.loc[0,'glu']=GLU
#     BP=bp_var.get()
#     DF.loc[0,'bp']=BP
#     SKIN=stn_var.get()
#     DF.loc[0,'skin']=SKIN
#     INSULIN=insulin_var.get()
#     DF.loc[0,'insulin']=INSULIN
#     MASS=bmi_var.get()
#     DF.loc[0,'mass']=MASS
#     PEDI=dpf_var.get()
#     DF.loc[0,'pedi']=PEDI
#     AGE=age_var.get()
#     DF.loc[0,'age']=AGE
#                 # print(DF.shape)
# DB=DF 
# def Output():
#     DB["Preg"] = pd.to_numeric(DB["Preg"])
#     DB["glu"] = pd.to_numeric(DB["glu"])
#     DB["bp"] = pd.to_numeric(DB["bp"])
#     DB["skin"] = pd.to_numeric(DB["skin"])
#     DB["insulin"] = pd.to_numeric(DB["insulin"])
#     DB["bmi"] = pd.to_numeric(DB["bmi"])
#     DB["pedi"] = pd.to_numeric(DB["pedi"])
#     DB["age"] = pd.to_numeric(DB["age"])
    
# Output=model.predict(DB)
# if Output==1:
#     result='Diabetic'
# elif Output==0:
#     result='Non-Diabetic' 



 

class diabetes:
    def __init__(self, window):
        self.window=window
        self.window.geometry("1530x790+0+0") #window creation
        self.window.title("diabetic prediction") #title of project
            
        lbl_title=Label(self.window,text="DIABETES PREDICTION  SYSTEM", font=("times new roman",37,"bold"),fg="red", bg="skyblue")
        lbl_title.place(x=0, y=0, width=1530,height=50)  # measurement and display on screen 
            
        preg_var=tk.StringVar() 
        glu_var=tk.StringVar() 
        bp_var=tk.StringVar()
        stn_var=tk.StringVar()
        insulin_var=tk.StringVar()
        bmi_var=tk.StringVar()
        dpf_var=tk.StringVar()
        age_var=tk.StringVar()
        
        
            # image frame 
        img_frame=Frame(self.window, bd=2,relief=RIDGE, bg="white") 
        img_frame.place(x=0,y=50, width=1530,height=180)

            # 1st image
        img1=Image.open("doc1.jpg")      
        img1=img1.resize((540,180),Image.ANTIALIAS)
        self.photo1=ImageTk.PhotoImage(img1)

        self.img_1=Label(img_frame,image=self.photo1)
        self.img_1.place(x=0,y=0,width=540,height=180)

            # 2nd image
        img2=Image.open("doc2.webp")      
        img2=img2.resize((540,180),Image.ANTIALIAS)
        self.photo2=ImageTk.PhotoImage(img2)

        self.img_2=Label(img_frame,image=self.photo2)
        self.img_2.place(x=540,y=0,width=540,height=180)

            # 3rd image
        img3=Image.open("DOC3.webp")      
        img3=img3.resize((540,180),Image.ANTIALIAS)
        self.photo3=ImageTk.PhotoImage(img3)

        self.img_3=Label(img_frame,image=self.photo3)
        self.img_3.place(x=1000,y=0,width=540,height=180)
            
        img4=Image.open("who1.webp")      
        img4=img4.resize((263,452),Image.ANTIALIAS)
        self.photo4=ImageTk.PhotoImage(img4)
            #  mainframe
        main_frame=Frame(self.window, bd=2,relief=RIDGE, bg="white") 
        main_frame.place(x=10,y=240, width=1520,height=560)
        

            # upper frame
        upper_frame=LabelFrame(main_frame, bd=2,relief=RIDGE,text="Enter Your Data",font=("times new roman",20,"bold"),fg="red", bg="orange") 
        upper_frame.place(x=20,y=6, width=700,height=520)
            # side frame
        side_frame=LabelFrame(main_frame, bd=2, bg="light green",text="Your Diabetes Prediction Result",font=("times new roman",20,"bold"),fg="red") 
        side_frame.place(x=600,y=6, width=95000,height=520)
            
            # button
            
            
        lbl_preg=Label(upper_frame,font=("arial",16,"bold"),text="Pregnancies:",bg="orange")
        lbl_preg.grid(row=1,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_preg=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_preg.grid(row=1,column=1,padx=2,pady=7)
            
            # +++
        lbl_glu=Label(upper_frame,font=("arial",16,"bold"),text="Glucose :",bg="orange")
        lbl_glu.grid(row=2,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_glu=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_glu.grid(row=2,column=1,padx=2,pady=7)
            
            # *******---
        lbl_bp=Label(upper_frame,font=("arial",16,"bold"),text="Blood Pressure:", bg="orange")
        lbl_bp.grid(row=3,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_bp=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_bp.grid(row=3,column=1,padx=2,pady=7)
            
            # **------//////
        lbl_stn=Label(upper_frame,font=("arial",16,"bold"),text="Skin-Thickness:", bg="orange")
        lbl_stn.grid(row=4,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_stn=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_stn.grid(row=4,column=1,padx=2,pady=7)
            
            # ********-------------/////
        lbl_insulin=Label(upper_frame,font=("arial",16,"bold"),text="Insulin:", bg="orange")
        lbl_insulin.grid(row=5,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_insulin=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_insulin.grid(row=5,column=1,padx=2,pady=7)
            
            # +++++++++*********------------
        lbl_bmi=Label(upper_frame,font=("arial",16,"bold"),text="BMI:", bg="orange")
        lbl_bmi.grid(row=6,column=0,padx=2,pady=7,sticky=W)
            
        
        txt_bmi=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_bmi.grid(row=6,column=1,padx=2,pady=7)
            
            # ++++++++++-------**********/////////////////
        lbl_dpf=Label(upper_frame,font=("arial",16,"bold"),text="DiabetesPedigreeFunction:", bg="orange")
        lbl_dpf.grid(row=7,column=0,padx=2,pady=7,sticky=W)
        
        
        txt_dpf=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_dpf.grid(row=7,column=1,padx=2,pady=7)
            
            # ++++++++++-------------////////////***********
        lbl_age=Label(upper_frame,font=("arial",16,"bold"),text="Age:", bg="orange")
        lbl_age.grid(row=8,column=0,padx=2,pady=7,sticky=W)
        
        
        txt_age=ttk.Entry(upper_frame,width=22,font=("arial",16))
        txt_age.grid(row=8,column=1,padx=2,pady=7)
            
            
            
        self.img_4=Label(side_frame,image=self.photo4)
        self.img_4.place(x=650,y=0,width=263,height=452)
            
            
        welcome=Label(side_frame,text="Take Care",font=("Times new roman",20, "bold"),fg="red", bg="lightblue")
        welcome.place(x=0,y=450,width=950,height=40)
        
        Predict_entrybox=Button(side_frame,text="Predict",font=("arial",15, "bold"),width=30,bg="blue",fg="white")
        Predict_entrybox.grid(row=0, column=0,padx=2,pady=5)
                            
        txt_predict=ttk.Entry(side_frame,width=30,font=("arial",16))
        txt_predict.grid(row=4,column=0,padx=2,pady=2)
        # 
        
           





   
        
        
        
        
    # prediction button
        
        
        #closing window
if __name__=="__main__":
    window=Tk()
    obj=diabetes(window)
    window.mainloop()