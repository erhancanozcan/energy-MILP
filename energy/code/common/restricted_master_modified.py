from gurobipy import *
import numpy as np

#A Computational Study of Dantzig-Wolfe Decomposition
class restricted_master:
    
    def __init__(self,num_homes,horizon,Q,D_P,D_d_list):
        
        self.num_homes=num_homes
        self.horizon=horizon
        self.Q=Q
        #D_P, and D_d are feature extractors from the extreme point.
        self.D_P=D_P
        self.D_d_list=D_d_list
        
        self.extreme_points=[None]*num_homes
        self.extreme_rays=[None]*num_homes
        
        self.num_e_p_per_home=[0]*num_homes
        self.num_e_r_per_home=[0]*num_homes
        
        
        #self.extractor=np.array([1,1,0,0])
        
        self.lambdas=[None]*num_homes
        self.thetas=[None]*num_homes
        
        #To DO: Objective Development Modify the variables below.
        self.obj_e_p_coeff=[None]*num_homes
        self.obj_e_r_coeff=[None]*num_homes
        
        #To DO: Constraint Development Modify the variables below.
        self.constr_e_p_coeff=[None]*num_homes
        self.constr_e_r_coeff=[None]*num_homes
        
        self.prob=Model("r_m")
        self.prob.Params.Threads=4
        
        self.s=self.prob.addVars(horizon,lb=0,name="s")
        self.a=self.prob.addVars(horizon,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="a")
        
        self.pos=self.prob.addConstrs((self.s[k]-self.a[k]>=0 for k in range(horizon)))
        self.neg=self.prob.addConstrs((self.s[k]+self.a[k]>=0 for k in range(horizon)))
        
        self.objective=[]
        self.iter_num=1
        

    
    def add_first_extreme_points(self,e_p,i):
        self.lambdas[i]=[self.prob.addVar(lb=0,name="l"+str(i)+"_0")]
        self.extreme_points[i]=[e_p]
        self.prob.addConstr(quicksum(self.lambdas[i][idx] for idx in range(len(self.lambdas[i])))==1,name="sum_lambda"+str(i))
    
    def add_first_extreme_rays(self,e_r,i):
        self.thetas[i]=[self.prob.addVar(lb=0,name="t"+str(i)+"_0")]
        self.extreme_rays[i]=[e_r]
            
    def add_other_extreme_points(self,e_p,i):
        num=len(self.lambdas[i])
        self.lambdas[i].append(self.prob.addVar(lb=0,name="l"+str(i)+"_"+str(num)))
        #self.lambdas[i]=self.lambdas[i]+1
        #NEW
        #NEW
        self.extreme_points[i].append(e_p)
        delete_this=self.prob.getConstrByName("sum_lambda"+str(i))
        self.prob.remove(delete_this)
        #self.prob.addConstr(quicksum(self.lambdas[i][idx] for idx in range(len(self.lambdas[i])))==1,name="sum_lambda"+str(i))
        if self.iter_num>3:
        #    #print("write a constraint by using self.ex_columns_list")
            self.ex_columns_list[i].append(len(self.lambdas[i])-1)
            self.prob.addConstr(quicksum(self.lambdas[i][self.ex_columns_list[i][idx]] for idx in range(len(self.ex_columns_list[i])))==1,name="sum_lambda"+str(i))
        else:
            self.prob.addConstr(quicksum(self.lambdas[i][idx] for idx in range(len(self.lambdas[i])))==1,name="sum_lambda"+str(i))
        
        #self.prob.chgCoeff(delete_this,self.lambdas[i][-1],1.0)
                
    def add_other_extreme_rays(self,e_r,i):
        num=len(self.thetas[i])
        self.thetas[i].append(self.prob.addVar(lb=0,name="t"+str(i)+"_"+str(num)))
        #self.thetas[i]=self.thetas[i]+1
        self.extreme_rays[i].append(e_r)
    
    def coupling_constrs(self):
        #print('can')
        for k in range(self.horizon):
            constr=self.prob.getConstrByName("coupling_"+str(k))
            if constr==None:
                LHS=LinExpr()
                LHS.add(self.a[k], 1)
                
                for i in range(self.num_homes):
                    j= len(self.extreme_points[i])-1
                    
                    coeff=np.dot(self.D_P[i][k,:],self.extreme_points[i][j])
                    self.constr_e_p_coeff[i]=[coeff]
                    
                    LHS.add(self.lambdas[i][j], coeff)
                    
                ### TO DO you may need to write this loop for an extreme ray!
                
                self.prob.addLConstr(LHS, '=', self.Q[k], name="coupling_"+str(k))
                
                self.prob.update()  
            
            else:
                
                for i in range(self.num_homes):
                    j= len(self.extreme_points[i])-1 
                    #print(j- len(self.lambdas[i]))
                    coeff=np.dot(self.D_P[i][k,:],self.extreme_points[i][j])
                    self.constr_e_p_coeff[i]=[coeff]
                    if self.iter_num>3:
                        if j in self.ex_columns_list[i]:
                            self.prob.chgCoeff(constr,self.lambdas[i][j],coeff)
                    else:
                         self.prob.chgCoeff(constr,self.lambdas[i][j],coeff)   
            
            self.prob.update() 
                
                

        if self.iter_num==1:
            
            for i in range(self.num_homes):
                j= len(self.extreme_points[i])-1
                coeff=np.dot(self.D_d_list[i],self.extreme_points[i][j])
                
                if self.obj_e_p_coeff[i]==None:
                    self.obj_e_p_coeff[i]=[coeff]
                else:
                    self.obj_e_p_coeff[i].append(coeff)
                    
            
            """
            TO DO: #Extreme rays in objective function are missing!
            """
            
            homes_with_extreme_rays=[]
            
            for i in range(self.num_homes):
                if self.extreme_rays[i]!=None:
                    homes_with_extreme_rays.append(i)
                    
            
            self.prob.setObjective(quicksum(self.s[k] for k in range(self.horizon))+\
                                   quicksum(self.lambdas[i][j]*self.obj_e_p_coeff[i][j] for i in range(self.num_homes) 
                                                                for j in range (len(self.extreme_points[i])))+\
                                   quicksum(self.thetas[i][j]*self.obj_e_r_coeff[i][j] for i in homes_with_extreme_rays
                                                                                        for j in range(len(self.extreme_rays[i]))),GRB.MINIMIZE)
            self.prob.update()
        
        else:
            #print("can")
            obj=self.prob.getObjective()
            
            for i in range(self.num_homes):
                j= len(self.extreme_points[i])-1
                
                coeff=np.dot(self.D_d_list[i],self.extreme_points[i][j])
                self.obj_e_p_coeff[i].append(coeff)
                
                #obj.addTerms(coeff, self.lambdas[i][j])
                if self.iter_num>3:
                    if j in self.ex_columns_list[i]:
                        self.lambdas[i][j].Obj=coeff
                else:
                    self.lambdas[i][j].Obj=coeff
            
            
            """
            TO DO: #Extreme rays in objective function are missing!
            """
            #self.prob.setObjective(obj,GRB.MINIMIZE)
            self.prob.update()
        
        
        self.iter_num+=1
            
                
                
                
            # if constr!=None:
            #     self.prob.remove(constr)
            # self.prob.update()
            # extreme_point=0
            
            
            
            # LHS=LinExpr()
            # LHS.add(self.a[k], 1)
            
            
            # #e_c=[]
            # for i in range(self.num_homes):
            #     for j in range(len(self.extreme_points[i])):
            #         coeff=np.dot(self.D_P[i][k,:],self.extreme_points[i][j])
            #         LHS.add(self.lambdas[i][j], coeff)
            

            # for i in range(self.num_homes):
            #     if self.extreme_rays[i] != None:
            #         for j in range(len(self.extreme_rays[i])):
            #             coeff=np.dot(self.D_P[i][k,:],self.extreme_rays[i][j])
            #             LHS.add(self.thetas[i][j], coeff)
            
            # self.prob.addLConstr(LHS, '=', self.Q[k], name="coupling_"+str(k))
            # #self.prob.addConstr(LHS, '==', self.Q[k], name="coupling_"+str(k))
            
            # #self.coupling=self.prob.addConstr((LHS== 10),name="coupling")
            # self.prob.update()   
        #print("can")
        
        # for i in range(self.num_homes):
        #     # for j in range(len(self.extreme_points[i])):
        #     #     coeff=np.dot(self.D_d_list[i],self.extreme_points[i][j])
                
        #     #     if self.obj_e_p_coeff[i]==None:
        #     #         self.obj_e_p_coeff[i]=[coeff]
        #     #     else:
        #     #         self.obj_e_p_coeff[i].append(coeff)
        #     j= len(self.extreme_points[i])-1
        #     coeff=np.dot(self.D_d_list[i],self.extreme_points[i][j])
            
        #     if self.obj_e_p_coeff[i]==None:
        #         self.obj_e_p_coeff[i]=[coeff]
        #     else:
        #         self.obj_e_p_coeff[i].append(coeff)
            
            
        #     if self.extreme_rays[i] != None:
        #         j=len(self.extreme_rays[i])-1
        #         coeff=np.dot(self.D_d_list[i],self.extreme_rays[i][j])
        #         if self.obj_e_r_coeff[i]==None:
        #             self.obj_e_r_coeff[i]=[coeff]
        #         else:
        #             self.obj_e_r_coeff[i].append(coeff)
                
            
        # """
        # TO DO: #Extreme rays in objective function are missing!
        # """
        
        # homes_with_extreme_rays=[]
        
        # for i in range(self.num_homes):
        #     if self.extreme_rays[i]!=None:
        #         homes_with_extreme_rays.append(i)
        
        
        
        # self.prob.setObjective(quicksum(self.s[k] for k in range(self.horizon))+\
        #                        quicksum(self.lambdas[i][j]*self.obj_e_p_coeff[i][j] for i in range(self.num_homes) 
        #                                                     for j in range (len(self.extreme_points[i])))+\
        #                        quicksum(self.thetas[i][j]*self.obj_e_r_coeff[i][j] for i in homes_with_extreme_rays
        #                                                                             for j in range(len(self.extreme_rays[i]))),GRB.MINIMIZE)
        # self.prob.update()
        # #self.prob.getRow(self.coupling)
