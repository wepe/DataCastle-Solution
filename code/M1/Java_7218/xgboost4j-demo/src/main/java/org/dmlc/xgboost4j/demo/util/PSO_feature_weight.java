package org.dmlc.xgboost4j.demo.util;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.sql.Date;
import java.util.ArrayList;
import java.util.HashMap;

import org.dmlc.xgboost4j.DMatrix;





import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.supportVector.*;




//适应值
/*class Fit{
	float[][][] predicts;
	float[] labels;

//	double fit_value;  //适应值
	
	
	
	//cfit1构造函数，初始化分类器、训练集和测试集
	public Fit(float[][][] predicts,float[] labels){
		this.predicts=predicts;
		this.labels=labels;
	}

	//计算适应值方法2,基分类器权重
	double cfit2(double base_classifier_weight[]){
		float ensembleResult[][]=Base_fuction.caculate_ensemble_result(predicts, base_classifier_weight);
		AUCEval ae=new AUCEval();
		double result=ae.eval(ensembleResult, labels);
		return result;
	}

}*/




class PSO{
	int itera_mum; //最大迭代次数
	int dim; //维度
	double min_boundary[]; //每个维度的最小边界
	double max_boundary[];  //每个维度的最大边界
	int par_num;  //粒子群中粒子数量
	Particle par[];  //粒子群
	double g_best[]; //群体所有微粒经历过的最好位置
	double fit_g_best; //群体所有微粒经历过的最好位置时的适应值
	
	double w;  //惯性权重
	double c1; //常量，控制每个微粒推向p_best位置的统计加速项的权重
	double c2;	//常量，控制每个微粒推向g_best位置的统计加速项的权重
	
//	Fit f; //计算适应值的一个类
	
	float[][][] predicts;
	float[] labels;
	
	
	
	
	public double caculate_fit(double position[]){
		int base_classifier_num=predicts.length;
		int ins_num=predicts[0].length;
		float ensembleResult[][]=new float[ins_num][1];		
		for(int i = 0;i <ins_num;i++){
			ensembleResult[i][0]=0;
			for(int j = 0;j <base_classifier_num;j++){
				ensembleResult[i][0]+=predicts[j][i][0]*position[j];
			}	
		}
		
		AUCEval ae=new AUCEval();
		return ae.evalAUC(ensembleResult, labels);
		
					
	}
	
	PSO(int itera_num,int dim,double min_boundary[],double max_boundary[],int par_num, double w, double c1, double c2,float[][][] predicts,float[] labels){
		this.itera_mum=itera_num;
		this.dim=dim;
		this.min_boundary=min_boundary;
		this.max_boundary=max_boundary;
		this.par_num=par_num;
		
		this.predicts=predicts;
		this.labels=labels;
		
		//先读入answer数据
		
		
		this.par=new Particle[this.par_num];
		this.g_best=new double [dim];
		for(int i=0;i<par_num;i++){
			//System.out.println("初始化粒子  "+i);
			par[i]=new Particle(dim, min_boundary, max_boundary);
		}
		intial_g_best();
		print_g_best();
		
		this.w=w;
		this.c1=c1;
		this.c2=c2;
		
		
	

	}
	
	//初始化g_best和其适应值
	void intial_g_best(){
		fit_g_best=par[0].fit_p_best;
		g_best=PSO_feature_weight.copy(par[0].p_best);
		for(int i=1;i<par_num;i++){
			if(par[i].fit_p_best>fit_g_best){
				g_best=PSO_feature_weight.copy(par[i].p_best);
				fit_g_best=par[i].fit_p_best;
			}
		}
	}
	
	//更新g_best和其适应值
	void update_g_best(){
		for(int i=0;i<par_num;i++){
			if(par[i].fit_p_best>fit_g_best){
				g_best=PSO_feature_weight.copy(par[i].p_best);
				fit_g_best=par[i].fit_p_best;
			}
		}
	}
	
	void print_g_best(){
		System.out.println("g_best\t"+fit_g_best+"\t");
		for(int i=0;i<dim;i++){
			System.out.print(g_best[i]+" ");
		}
		System.out.println();
	}
	
	//粒子群演化过程，返回最优解g_best[]
	public double[] evolove(){
		int t=0;
		while(t<itera_mum){
			t++;
			System.out.println("第"+t+"代演化");
			for(int j=0;j<par_num;j++){
				par[j].update_position(g_best, c1, c2, w);
				//System.out.println("更新位置  "+j);
				par[j].update_p_best();
			}
			update_g_best();
			print_g_best();
			
		}
		return g_best;
	}

	
	//粒子
	class Particle{
		int dim; //维度
		double min_boundary[]; //每个维度的最小边界
		double max_boundary[];  //每个维度的最大边界
		
		double position[]; //当前位置
		double fit_p; //当前适应值
		double velocity[]; //加速度
		double p_best[]; //个体的最佳位置
		double fit_p_best; //个体的最佳位置时的适应值
		
		
		Particle(int dim,double min_boundary[],double max_boundary[]) {
			this.dim=dim;
			this.min_boundary=min_boundary;
			this.max_boundary=max_boundary;
			
			position=new double [dim];
			velocity=new double [dim];
			for(int i=0;i<this.dim;i++){
				
				position[i]=Base_fuction.random_double(min_boundary[i],max_boundary[i],1);
				//选择每个维度距边界的最小值
				double v_boundary=(max_boundary[i]-min_boundary[i])/3;
				//double v_boundary=((position[i]-min_boundary[i])<(max_boundary[i]-position[i]))?(position[i]-min_boundary[i]):(max_boundary[i]-position[i]);
				
				velocity[i]=Base_fuction.random_double(-v_boundary,v_boundary,2);
			}
			fit_p=caculate_fit(position);
			p_best=PSO_feature_weight.copy(position);
			fit_p_best=fit_p;
		}
		
		
		//计算粒子的适应值
		//////////////////////////
		///！！！可更改多个计算方法
		//////////////////////////////
//		public double caculate_fit(Fit f){
//			//return f.cfit1(position);
//			return f.cfit2(position);
//		}
//		
		//更改位置
		void update_position(double g_best[],double c1,double c2,double w){
			//更改加速度velocity
			double r1,r2;
			for(int i=0;i<dim;i++){
				r1=Base_fuction.random_double(0, 1, 3);
				r2=Base_fuction.random_double(0, 1, 3);
				velocity[i]=w*velocity[i]+c1*r1*(p_best[i]-position[i])+c2*r2*(g_best[i]-position[i]);
			}
			//更改当前位置position
			for(int i=0;i<dim;i++){
				position[i]+=velocity[i];
				if(position[i]<min_boundary[i]){
					position[i]=min_boundary[i];
				}
				else if(position[i]>max_boundary[i]){
					position[i]=max_boundary[i];
				}
			}
		}
		
		//更改p_best及其适应值
		void update_p_best(){
			fit_p=caculate_fit(position);

			if(fit_p>fit_p_best){
				p_best=PSO_feature_weight.copy(position);
				fit_p_best=fit_p;
			}
		}

		void print_particle(){
			System.out.println(fit_p_best+"\t"+fit_p+"\t");
			for(int i=0;i<dim;i++){
				System.out.print(position[i]+" ");
			}
			System.out.println();
		}
	}

}




public class PSO_feature_weight {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	//dim维度=基分类器数量
	public static double[] excute(int dim,	float[][][] predicts,float[] labels) {
		int itera_mum=20; //最大迭代次数
		double min_boundary[]=new double[dim]; //每个维度的最小边界
		double max_boundary[]=new double[dim];  //每个维度的最大边界
		for(int i=0;i<dim;i++){
			min_boundary[i]=0.0;
			max_boundary[i]=2;
		}
		
		int par_num=20000;  //粒子群中粒子数量
		
		double w=0.5;  	//惯性权重
		double c1=0.5; 	//常量，控制每个微粒推向p_best位置的统计加速项的权重
		double c2=0.5;	//常量，控制每个微粒推向g_best位置的统计加速项的权重
		
		
		PSO p=new PSO(itera_mum, dim, min_boundary, max_boundary, par_num, w, c1, c2,predicts,labels);
		double base_classfier_weight[]=p.evolove(); 
		return base_classfier_weight;
	}
	
	//复制数组a
	static double[] copy(double a[]){
		double b[]=new double[a.length];
		for(int i=0;i<a.length;i++){
			b[i]=a[i];
		}
		return b;
	}

}
