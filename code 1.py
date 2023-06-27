'''-----------------imports-------------------'''
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import math as m
import os
import copy
from PIL import Image
from copy import *
'''-------------------variables-----------------'''

nTest=12629
Xmax=44
Ymax=44

#code copié
data_dir = 'C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée'
train_path = 'C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée/Train'
test_path = 'C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée/'

nb_train=6

image_data = []
image_labels = []

image_eloigne=[0,11,12,13,14,17,18,22,25,33]
image_proche=[0 for i in range(43)]

for i in range(nb_train):
    path = data_dir + '/Train/' + str(image_eloigne[i])
    images = os.listdir(path)
    print(i)
    for img in images:
        try:
            image = Image.open(path + '/' + img)
            resize_image =image.resize((Xmax,Ymax))
            image_data.append(np.asarray(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)
image_data = np.array(image_data)
image_labels = np.array(image_labels)

shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

'''-------------------classes-----------------'''

class kernel(): #un kernel
    def __init__(self,kx,ky,ix,iy,iz,r):
        if r:
            self.kx=kx
            self.ky=ky
            self.kz=iz
            self.ix=ix
            self.iy=iy
            self.iz=iz
            self.ox=ix-kx+1
            self.oy=iy-ky+1
            K=np.random.randn(kx,ky,iz)#matrice kernel
            self.K=K
            self.Biases=np.random.randn(self.ox,self.oy)

    def f(self,I):
        # return convolution(I,self.K)+self.Biases
        P=im2col(I,(self.kx,self.ky)).T
        F=np.reshape(self.K,self.kx*self.ky*self.kz,'F')
        B=self.Biases
        O=np.reshape(P.dot(F),(self.ox,self.oy),'F')
        return O
    def b(self,dzf,ep):
        Dzf=np.reshape(dzf,(self.kx,self.ky,self.iz))
        self.K-=ep*Dzf



        # s=0
        # for i in range(self.kx):
        #     for j in range(self.ky):
        #         for k in range(self.iz):
        #                 s+=self.K[i,j,k]
        # if s<-1 or s>1:
        #     self.K=self.K/s


class Ckernel(): #couche de kernels
    def __init__(self,n,kx,ky,ix,iy,iz,r=True): #dx,dy
        self.n=n
        self.kx=kx
        self.ky=ky
        self.ix=ix
        self.iy=iy
        self.iz=iz
        self.ox=ix-kx+1
        self.oy=iy-ky+1
        self.oz=n
        L=np.array([kernel(kx,ky,ix,iy,iz,r) for _ in range(self.n)])
        self.L=L #n*kx*ky*iz
        self.O=np.random.randn(self.ox,self.oy,self.oz)
        self.type="Kernel"

    def f(self,I):
        self.I=np.copy(I)
        O=[]
        for i in range(self.n):
            O.append(self.L[i].f(I))
        O=np.array(O)
        return np.transpose(O,(1,2,0))

    def b(self,dzy,ep):
        ox,oy,oz=np.shape(dzy)
        Dzy=np.zeros((ox*oy,oz))
        for i in range(oz):
            Dzy[:,i]=np.reshape(dzy[:,:,i],(ox*oy),'F')
        dzf=im2col(self.I,(self.kx,self.ky)).dot(Dzy)
        for i in range(oz):
            self.L[i].Biases-=ep*dzy[:,:,i]
            self.L[i].K-=ep*np.reshape(dzf[:,i],(self.kx,self.ky,self.iz),'F')
        dzx=np.zeros((self.ix,self.iy,self.iz))

        F=np.zeros((self.kx*self.ky*self.iz,self.n))
        for i in range(self.n):
            F[:,i]=np.reshape(self.L[i].K,self.kx*self.ky*self.iz,'F')
        U=Dzy.dot(F.T)
        for i in range(self.ix):
            for j in range(self.iy):
                for d in range(self.iz):
                    for u in invm(i,j,d,self.ix,self.iy,self.iz,self.kx,self.ky):
                        p,q=u
                        dzx[i,j,d]+=U[p,q]

        for i in range(self.n):
            self.L[i].b(dzf[:,i],ep)
        return dzx


class CReLu() :
    def __init__(self):
        self.type="ReLu"

    def f(self,I):
        self.I=np.copy(I)
        ix,iy,iz=np.shape(I)
        O=np.zeros((ix,iy,iz))
        m=0
        for x in range(ix):
            for y in range(iy):
                for z in range(iz):
                    if I[x,y,z]<0:
                        O[x,y,z]=0
                    else:
                        O[x,y,z]=I[x,y,z]
                    # if O[x,y,z]>m:
                    #     m=O[x,y,z]
        self.m=np.linalg.norm(O)
        # self.m=m
        return O/(self.m)

    def b(self,dzy,ep):
        ux,uy,uz=np.shape(self.I)
        dzx=np.zeros((ux,uy,uz))
        for i in range(ux):
            for j in range(uy):
                for k in range(uz):
                    if self.I[i,j,k]>0:
                        dzx[i,j,k]=dzy[i,j,k]
        return dzx

class Cnorm() :
    def __init__(self):
        self.type="norm"

    def f(self,I):
        self.m=np.linalg.norm(I)
        return I/(self.m)

    def b(self,dzy,ep):
        return dzy



class Csigm() :
    def __init__(self):
        self.type="Sigm"

    def f(self,I):
        ix,iy,iz=np.shape(I)
        O=np.zeros((ix,iy,iz))
        for x in range(ix):
            for y in range(iy):
                for z in range(iz):
                    if I[x,y,z]>-200.:
                        O[x,y,z]=1/(1+m.exp(-I[x,y,z]))
                    else :
                        O[x,y,z]=0.
        self.O=np.copy(O)
        return O

    def b(self,dzy,ep):
        ux,uy,uz=np.shape(self.O)
        dzx=np.zeros((ux,uy,uz))
        for i in range(ux):
            for j in range(uy):
                for k in range(uz):
                    dzx[i,j,k]=self.O[i,j,k]*(1-self.O[i,j,k])
        return dzx

class Csoftmax():
    def __init__(self): #dx,dy
        self.type="softmax"
    def f(self,I):
        self.I=np.copy(I)
        return softmax(I)
    def b(self,dzy,ep):
        dyx=derivsoft(softmax(self.I))
        dzx=np.dot(dzy.T,dyx)
        return dzy

class Cfullconnect():
    def __init__(self,n,ix,iy,iz,r=True): #dx,dy
        self.n=n
        self.ix=ix
        self.iy=iy
        self.iz=iz
        self.ox=1
        self.oy=1
        self.oz=n
        self.type="Fully connected"
        self.L=Ckernel(n,ix,iy,ix,iy,iz)
    def f(self,I):
        self.I=np.copy(I)
        O=self.L.f(I)
        O=np.reshape(O,self.n)
        return O
    def b(self,dzy,ep):
        dzx=self.L.b(np.reshape(dzy,(1,1,self.n),'F'),ep)
        return dzx


class Cpool():
    def __init__(self,ix,iy,iz,mx,my,typ="max"): #attention, mx my et mz doivent etre diviseurs des dimensions de l'input
        self.typ=typ
        self.ix=ix
        self.iy=iy
        self.iz=iz
        self.mx=mx
        self.my=my
        self.ox=ix//mx
        self.oy=iy//my
        self.type="Pooling"
    def f(self,I):
        self.I=np.copy(I)
        O=np.zeros((self.ox,self.oy,self.iz))
        for z in range(self.iz):
            for x in range(self.ox):
                for y in range(self.oy):
                    s=0
                    m=0
                    for i in range(self.mx):
                        for j in range(self.my):
                            s+=I[i+(x*self.mx),j+(y*self.my),z]
                            if I[i+(x*self.mx),j+(y*self.my),z]>m:
                                m=I[i+(x*self.mx),j+(y*self.my),z]
                    if self.typ=="max":
                        O[x,y,z]=m
                    else :
                        O[x,y,z]=s/(self.ox*self.oy)
        return O
    def b(self,dzy,ep):
        dzx=np.zeros((self.ix,self.iy,self.iz))
        gx,gy=self.ix//self.mx,self.iy//self.my
        for z in range(self.iz):
            for x in range(self.ox):
                for y in range(self.oy):
                    mi,mj=0,0
                    m=0
                    for i in range(self.mx):
                        for j in range(self.my):
                            if self.I[i+(x*self.mx),j+(y*self.my),z]>m:
                                m=self.I[i+(x*self.mx),j+(y*self.my),z]
                                mi,mj=i+(x*self.mx),j+(y*self.my)
                    dzx[mi,mj,z]=dzy[x,y,z]
        return dzx

class loss():
    def _init_(self):
        self.type="loss"


class reseau():
    def __init__(self,X=1):
        self.L=[]

    def result(self,I):
        I=I
        for couche in self.L:
            # print(I)
            A=np.copy(I)
            I=couche.f(A)
        #il faut que la derniere couche soit full connected pour que I soit un vecteur
        return I

    def apprentissage(self,I,t,coef):
        I=I
        for couche in self.L:
            A=np.copy(I)
            I=couche.f(A)
        x=np.copy(I)
        #il faut que la derniere couche soit full connected pour que I soit un vecteur
        z=loss(x,t)
        print(z)
        dzx=x.copy()-t.copy() #dz/dx
        n=len(self.L)
        for i in range(n-1,-1,-1):
            cdzx=np.copy(dzx)
            dzx=self.L[i].b(cdzx,coef)








'''-------------------fonctions-----------------'''

# def convolution2(I,K,b=True):
#     ix,iy,iz=np.shape(I)
#     kx,ky,kz=np.shape(K)
#     O=np.zeros((ix-kx+1,iy-ky+1,iz))
#     if b:
#         for z in range(iz):
#             for x in range(ix-kx+1):
#                 for y in range(iy-ky+1):
#                     for i in range(kx):
#                         for j in range(ky):
#                             O[x,y,z]+=I[x+i,y+j,z]*K[i,j,z]
#     return O

def convolution(I,K,b=True):
    ix,iy,iz=np.shape(I)
    kx,ky,kz=np.shape(K)
    O=np.zeros((ix-kx+1,iy-ky+1))
    I=np.array(I)
    K=np.array(K)
    if b:
        for x in range(ix-kx+1):
            for y in range(iy-ky+1):
                O[x,y]=np.sum(I[x:(kx+x),y:(y+ky),:]*K)
    return O

def loss(x,t): #x est le resultat, t est la cible
    #on suppose x normé
    return (1/2)*(np.linalg.norm(x-t)**2)

def im2col(I,t):
    dim=np.ndim(I)
    kx,ky=t
    if dim==2 :
        ix,iy=np.shape(I)
        ligne=ix-kx+1
        colonne=iy-ky+1
        B=np.zeros((kx*ky,colonne*ligne))
        i=0
        for y in range(colonne):
            for x in range(ligne):
                B[:,i]=np.reshape(I[x:(kx+x),y:(ky+y)],kx*ky,'F')
                i+=1
    elif dim==3 :
        ix,iy,iz=np.shape(I)
        ligne=ix-kx+1
        colonne=iy-ky+1
        B=np.zeros((kx*ky*iz,colonne*ligne))
        i=0
        for y in range(colonne):
            for x in range(ligne):
                B[:,i]=np.reshape(I[x:(kx+x),y:(ky+y),:],kx*ky*iz,'F')
                i+=1
    return B

def invm(i,j,d,ix,iy,iz,kx,ky):
    L=[]
    for y in range(ky):
        for x in range(kx):
            px=x-kx+1+i
            py=y-ky+1+j
            if px>=0 and py>=0 and px+kx<=ix and py+ky<=iy:
                p=px+py*(ix-kx+1)
                q=d*kx*ky+(ky-y-1)*kx+kx-x-1
                L.append((p,q))
    return L

def softmax(I):
    n=len(I)
    a=0
    for i in range(n):
        a+=m.exp(I[i])
    for i in range(n):
        I[i]=m.exp(I[i])/a
    return I

def derivsoft(x):
    n=len(x)
    M=np.zeros((n,n))
    for i in range(n):
        M[:,i]=np.copy(x)
    return M*(np.identity(n)-M.T)


def lesimages():

    for i in range(nb_train):
        path = data_dir + '/Train/' + str(i)
        images = os.listdir(path)
        print(i)
        for img in images:
            try:
                image = Image.open(path + '/' + img)
                resize_image =image.resize((Xmax,Ymax))
                image_data.append(np.asarray(resize_image))
                image_labels.append(i)
            except:
                print("Error in " + img)
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    shuffle_indexes = np.arange(image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    image_data = image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]







def superaprentissage(R,size,maxepoch,test_size,coef):

    for i in range(maxepoch):
        print("epoch : "+str(i))
        for j in range(size):
            print(str(j)+'/'+str(size))
            img=image_data[j]
            t=np.zeros(nb_train)
            t[image_labels[j]]=1
            R.apprentissage(img,t,coef)
    somme=0
    print("--testing--")
    for i in range(test_size):
        l=R.result(image_data[i+size]).copy()
        ma=0
        x=0
        n=len(l)
        for j in range(n):
            if l[j]>ma:
                x=j
                ma=l[j]
        print(str(x)+" ? "+str(image_labels[i+size])+" i= "+str(i))
        print(l)
        if x==image_labels[i+size]:
            somme+=1
    return somme/test_size



def supersuperaprentissage(L,size,maxepoch,test_size,coef):
    nn=len(L)
    for i in range(maxepoch):
        print("epoch : "+str(i))

        for j in range(size):
            print(str(j)+'/'+str(size))
            img=image_data[j]
            t=np.zeros(nb_train)
            t[image_labels[j]]=1
            R.apprentissage(img,t,coef)
    somme=0
    print("--testing--")
    for i in range(test_size):
        l=R.result(image_data[i+size]).copy()
        ma=0
        x=0
        n=len(l)
        for j in range(n):
            if l[j]>ma:
                x=j
                ma=l[j]
        print(str(x)+" ? "+str(image_labels[i+size])+" i= "+str(i))
        print(l)
        if x==image_labels[i+size]:
            somme+=1
    return somme/test_size




lereseau=reseau()
lereseau.L=[Ckernel(4,3,3,44,44,3),CReLu(),Ckernel(4,3,3,42,42,4),CReLu(),Cpool(40,40,4,2,2),Ckernel(8,3,3,20,20,4),CReLu(),Ckernel(8,3,3,18,18,8),CReLu(),Cpool(16,16,8,2,2),Ckernel(16,3,3,8,8,8),CReLu(),Ckernel(16,3,3,6,6,16),CReLu(),Cfullconnect(nb_train,4,4,16),Cnorm()]
# lereseau.L=[Cfullconnect(nb_train,Xmax,Ymax,3),Cnorm()]
# lereseau.L=[Ckernel(6,3,3,44,44,3),CReLu(),Ckernel(12,3,3,42,42,6),CReLu(),Cfullconnect(nb_train,40,40,12),Cnorm()]
# lereseau.L=[Ckernel(4,3,3,5,5,3),CReLu(),Cfullconnect(nb_train,3,3,4),Cnorm(),Csoftmax()]


# print(superaprentissage(lereseau,400,20,200,10))
# print(superaprentissage(lereseau,10,5,20))
# superaprentissage(lereseau,1,5,2)

'''-------------------main-----------------'''








#
#
# img1 = Image.open("C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée/Test/00002.png")
# img1=img1.resize((3,3))
# img1=np.asarray(img1)
#
# img2 = Image.open("C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée/Test/00093.png")
# img2=img2.resize((3,3))
# img2=np.asarray(img2)

#
# img1=np.array([[[1,1,1],[0,0,0],[1,1,1]],[[2,2,2],[0,0,0],[2,2,2]],[[1,1,1],[0,0,0],[1,1,1]]])
# img2=np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[2,2,2],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])

# print(lereseau.result(img1))
#
#
# t1=np.zeros(nb_train)
# t1[0]=1
# t2=np.zeros(nb_train)
# t2[1]=1
# for i in range(100):
#     lereseau.apprentissage(img1,t1,0.05)
#     print(lereseau.result(img1))
#     print("-----______-----______-----______------______------______------______-----______-----")
#     lereseau.apprentissage(img2,t2,0.05)
#     print(lereseau.result(img2))
#
#
# print(lereseau.result(img1))
# print(lereseau.result(img2))
# plt.imshow(img1.astype(np.uint8))
# plt.show()

#
#
# img = Image.open("C:/Users/Propriétaire/Documents/TIPE 2023/base de donnée/Test/00001.png")
#
# img=img.resize((44,44))
#
# re=Csigm()
# ke=Ckernel(1,3,3,44,44,3)
#
# img=np.asarray(img)
# img=img
#
# K=np.array([[[[1,1,1],[0,0,0],[-1,-1,-1]],[[2,2,2],[0,0,0],[-2,-2,-2]],[[1,1,1],[0,0,0],[-1,-1,-1]]]])
# ke.L[0].K=np.transpose(K,(1,2,3,0))
#
# a=ke.f(img)
# img2=re.f(ke.f(img))
# print(img2[23,26])
# plt.imshow(img2.astype(np.uint8))
#
# plt.show()


