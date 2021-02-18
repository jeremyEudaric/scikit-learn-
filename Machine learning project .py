import numpy as np
import glob
from sklearn import neighbors
import sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix






f0_files=glob.glob('C:/Users/jejeu/OneDrive/Bureau/TP_Social_Robotics/BabyEars_data/*.f0')
en_files=glob.glob('C:/Users/jejeu/OneDrive/Bureau/TP_Social_Robotics/BabyEars_data/*.en')


f0_database=np.array([[]])
en_database=np.array([[]])
database_target=np.array([])

for f0_file in f0_files:
    f0_sample=np.loadtxt(f0_file)

    if 'ap' in f0_file:
        file_class='ap'
    elif 'pr' in f0_file:
        file_class='pw'
    elif 'at' in f0_file:
        file_class='at'

    if file_class=='ap' or file_class=='pw' or file_class =='at':
        local_derivative=(f0_sample[1:,1]-f0_sample[:-1,1])-(f0_sample[1:,0]-f0_sample[:-1,0])
        f0_wo_zeros_mask=f0_sample[:,1]!=0
        f0_sample=f0_sample[f0_wo_zeros_mask,:]

        database_target=np.concatenate((database_target,np.array([file_class])))

        mean_functional=np.mean(f0_sample[:,1])
        max_functional=np.max(f0_sample[:,1])
        range_functional=np.max(f0_sample[:,1])-np.min(f0_sample[:,1])
        variance=np.var(f0_sample[:,1])
        median=np.median(f0_sample[:,1])
        first_quartile=np.quantile(f0_sample[:,1],0.25)
        third_quartile=np.quantile(f0_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        if f0_database.shape[1]==0:
            f0_database=np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])
        else:
            f0_database=np.concatenate((f0_database,np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])),axis=0)
        en_file=f0_file[:-2]+'en'
        en_sample=np.loadtxt(en_file)

        local_derivative = (en_sample[1:,1]-en_sample[:-1,1])-(en_sample[1:,0]-en_sample[:-1,0])
        en_sample=en_sample[f0_wo_zeros_mask,:]

        mean_functional=np.mean(en_sample[:,1])
        max_functional=np.mean(en_sample[:,1])
        range_functional=np.max(en_sample[:,1])-np.min(en_sample[:,1])
        variance=np.var(en_sample[:,1])
        median=np.median(en_sample[:,1])
        first_quartile=np.quantile(en_sample[:,1],0.25)
        third_quartile=np.quantile(en_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        if en_database.shape[1]==0:
            en_database=np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])
        else:
            en_database=np.concatenate((en_database,np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])),axis=0)

    








baby_ears_database=np.concatenate((f0_database,en_database),axis=1)







################### Build training database and a testing database#################################

n_neighbors = 2


# Create KNN classifer
knn1 = neighbors.KNeighborsClassifier(n_neighbors)
x1= baby_ears_database
y1= database_target



# split data set an training 70% trainning
x1_train , x1_test, y1_train, y1_test = sklearn.model_selection.train_test_split(x1,y1, test_size = 0.4,random_state = 1, stratify=y1)



#Standardize data
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_transforme = scaler.transform(x1_train)
x1_test_transforme = scaler.transform(x1_test)



# train the model
knn1.fit(x1_train,y1_train)

y_pred1 = knn1.predict(x1_test)


confusion_matrix1=metrics.confusion_matrix(y1_test,y_pred1,labels=['ap','pw','at'])
accuracy=np.sum(np.diag(confusion_matrix1))/np.sum(confusion_matrix1)
print('accuracy: ',accuracy)

print('confusion matrix baby :\n',confusion_matrix1)


# To define the best 'n_neighbors that me must use for this model 
#param_grid={'n_neighbors' : [1,2,3,5,7,10]}
#grid_search =  GridSearchCV(knn, param_grid, cv =5 ,n_jobs = -1)
#grid_search.fit(x_train,y_train )
#print('le meilleur  K est : ' , grid_search.best_params_)



############################################################################################


f0_files=glob.glob('C:/Users/jejeu/OneDrive/Bureau/TP_Social_Robotics/Kismet_data_intent/*.f0')
en_files=glob.glob('C:/Users/jejeu/OneDrive/Bureau/TP_Social_Robotics/Kismet_data_intent/*.en')


database=[]
database_target=[]

for f0_file in f0_files:
    f0_sample=np.loadtxt(f0_file)

    if 'ap' in f0_file[-5:-3]: # -8:-6 for baby_ears dataset
        file_class='ap'
    elif 'pw' in f0_file[-5:-3]: # -8:-6 for baby_ears dataset
        file_class='pw'
    elif 'at' in f0_file[-5:-3]: # -8:-6 for baby_ears dataset
        file_class='at'

    if file_class=='ap' or file_class=='pw' or file_class == 'at':
        local_derivative=(f0_sample[1:,1]-f0_sample[:-1,1])-(f0_sample[1:,0]-f0_sample[:-1,0])
        f0_wo_zeros_mask=f0_sample[:,1]!=0
        f0_sample=f0_sample[f0_wo_zeros_mask,:]

        database_target=np.concatenate((database_target,np.array([file_class])))

        mean_functional=np.mean(f0_sample[:,1])
        max_functional=np.max(f0_sample[:,1])
        range_functional=np.max(f0_sample[:,1])-np.min(f0_sample[:,1])
        variance_functional=np.var(f0_sample[:,1])
        median_functional=np.median(f0_sample[:,1])
        first_quartile=np.quantile(f0_sample[:,1],0.25)
        third_quartile=np.quantile(f0_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        f0_functionals = [mean_functional,max_functional,range_functional,variance_functional,median_functional,first_quartile,third_quartile,mean_absolute_local_derivative]

        en_file=f0_file[:-2]+'en'
        en_sample=np.loadtxt(en_file)

        local_derivative = (en_sample[1:,1]-en_sample[:-1,1])-(en_sample[1:,0]-en_sample[:-1,0])
        en_sample=en_sample[f0_wo_zeros_mask,:]

        mean_functional=np.mean(en_sample[:,1])
        max_functional=np.mean(en_sample[:,1])
        range_functional=np.max(en_sample[:,1])-np.min(en_sample[:,1])
        variance_functional=np.var(en_sample[:,1])
        median_functional=np.median(en_sample[:,1])
        first_quartile=np.quantile(en_sample[:,1],0.25)
        third_quartile=np.quantile(en_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        en_functionals = [mean_functional,max_functional,range_functional,variance_functional,median_functional,first_quartile,third_quartile,mean_absolute_local_derivative]

        database.append(np.concatenate((f0_functionals,en_functionals),axis=0))


    
database=np.array(database)




################### Build training database and a testing database#################################


n_neighbors2 = 2

# Create KNN classifer
knn2 = neighbors.KNeighborsClassifier(n_neighbors2)
x2= database
y2= database_target



# split data set an training 60% trainning
x2_train , x2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(x2,y2, test_size = 0.4,random_state = 1, stratify=y2)



#Standardize data
scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_transforme = scaler.transform(x2_train)
x2_test_transforme = scaler.transform(x2_test)



# train the model
knn2.fit(x2_train,y2_train)

y_pred2 = knn2.predict(x2_test)
#print(y_pred)


confusion_matrix2=metrics.confusion_matrix(y2_test,y_pred2,labels=['ap','pw','at'])
accuracy=np.sum(np.diag(confusion_matrix2))/np.sum(confusion_matrix2)
print('accuracy: ',accuracy)

print('confusion matrix kismet:\n',confusion_matrix2)


####################################### pooling Test sur baby ####################################

X_train = np.concatenate((x1_train,x2_train))
Y_train = np.concatenate((y1_train,y2_train))

    
knn1.fit(X_train,Y_train)

y_pred1 = knn1.predict(x1_test)


confusion_matrix1=metrics.confusion_matrix(y1_test,y_pred1,labels=['ap','pw','at'])
accuracy=np.sum(np.diag(confusion_matrix1))/np.sum(confusion_matrix1)
print('accuracy: ',accuracy)

print('confusion matrix pooling test dans baby :\n',confusion_matrix1)


####################################### pooling Test sur kiment  ####################################




knn2.fit(X_train,Y_train)

y_pred2 = knn2.predict(x2_test)
#print(y_pred)


confusion_matrix2=metrics.confusion_matrix(y2_test,y_pred2,labels=['ap','pw','at'])
accuracy=np.sum(np.diag(confusion_matrix2))/np.sum(confusion_matrix2)
print('accuracy: ',accuracy)

print('confusion matrix  pooling sur kimet:\n',confusion_matrix2)