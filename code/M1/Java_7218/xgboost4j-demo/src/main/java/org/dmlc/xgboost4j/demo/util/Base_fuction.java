package org.dmlc.xgboost4j.demo.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import weka.core.*;
import weka.classifiers.*;
import weka.core.converters.ArffSaver;

public class Base_fuction {
	
	Instances Ori_instances; //初始数据集
	
	Instances dataset_maj;  //多类样本数据集
	Instances dataset_min;  //少类样本数据集
	
	static String arff=".arff";
	static String txt=".txt";
	//String filepath; //读取文件路径
	//String filename; //读取文件名称
	
	
	//构造函数1:根据文件路径及名称初始化Ori_instances
	public Base_fuction(String filename,String filepath){
		try {
			BufferedReader br = new BufferedReader(new FileReader(filepath+filename+arff));
			Ori_instances = new Instances(br);
			if (Ori_instances.classIndex() == -1) 
				Ori_instances.setClassIndex(Ori_instances.numAttributes() - 1);				
		} 
		catch (Exception e) {
				e.printStackTrace();
		}
	}
	//构造函数2：将instances赋值给Ori_instances
	Base_fuction(Instances instances){
		Ori_instances=new Instances(instances);
	}
	
	//读数据
	public void GetData(String filename,String filepath){
		try {
			BufferedReader br = new BufferedReader(new FileReader(filepath+filename+arff));
			Ori_instances = new Instances(br);
			if (Ori_instances.classIndex() == -1) 
				Ori_instances.setClassIndex(Ori_instances.numAttributes() - 1);				
		} 
		catch (Exception e) {
				e.printStackTrace();
		}
	}
  //获取初始化数据集  	
	public Instances GetOri_instances(){
		return Ori_instances;
	}
	//样本集写入文件(txt文件)
	public void Outfile_instances_txt(){
			;
	}
		
	//样本集写入文件(arff文件)
	public static void Outfile_instances_arff(Instances instances,String filename, String filepath){
			
		//判断目录，若不存在则新建
        File file=new File(filepath);
        if  (!file .exists()  && !file .isDirectory())      
        {       
            System.out.println("//不存在"+filepath+"需创建");  
            file .mkdir();    
        }
        
		
		ArffSaver saver = new ArffSaver();
     	saver.setInstances(instances);
		String file_entire_name=filepath+filename+arff;
		try{
			saver.setFile(new File(file_entire_name));
			saver.writeBatch();
		}
		catch (Exception e){
			e.printStackTrace();
		}
			 
		System.out.println(file+"写入文件");
			
	}	
	
	
	//交叉分折,seed随机数种子，folds分折数,instances待拆分的数据集
	public static void partion_CV(int seed,int folds,Instances instances,String filename,String filepath){
		Random rand=new Random(seed);
		Instances rand_instances=new Instances(instances);
		rand_instances.randomize(rand);
		rand_instances.stratify(folds);
			
		for(int i=0;i<folds;i++){
			
			
			Instances train_instance=rand_instances.trainCV(folds, i);
			System.out.println("训练集"+i+":");
			print_instances_info(train_instance);
			
			String trainfile=filename+"_CV"+i+"_train";
			Outfile_instances_arff(train_instance,trainfile,filepath);
			
			Instances test_instance=rand_instances.testCV(folds, i);
			System.out.println("测试集"+i+":");				
			print_instances_info(test_instance);	
			
			String testfile=filename+"_CV"+i+"_test";
			Outfile_instances_arff(test_instance,testfile,filepath);
		}
	}	

	
	//拆分训练子集,seed随机数种子，sub_num子训练集数数,instances待拆分的数据集
	public static void partion_Sub(int seed,int sub_num,Instances instances,String filename,String filepath){
		Instances train_maj = new Instances(instances,0); //创建一个空数据集:多类训练集
		Instances train_min = new Instances(instances,0); //创建一个空数据集:少类训练集
		for(int j=0;j<instances.numInstances();j++){
			
			if(instances.instance(j).stringValue(instances.numAttributes()-1).toString().equals("0")){
				//第j个样本是多类
				train_maj.add(instances.get(j)); //数据集添加一个样本
			}
			else{
				//第j个样本是少类
				train_min.add(instances.get(j));
			}
		}
		System.out.println("多类训练集信息");
		Base_fuction.print_instances_info(train_maj);
		System.out.println("少类训练集信息");
		Base_fuction.print_instances_info(train_min);
		
		
		Random rand=new Random(seed);
		Instances rand_instances=new Instances(train_maj);
		rand_instances.randomize(rand);
		rand_instances.stratify(sub_num);
		
		for(int i=0;i<sub_num;i++){
			
			//一份多类和全部少类
			Instances train_sub=rand_instances.testCV(sub_num, i);	
			for(int j=0;j<train_min.numInstances();j++){
				train_sub.add(train_min.get(j));
			}
			
			System.out.println("训练子集"+i+":");
			print_instances_info(train_sub);
			
			String trainfile=filename+"_Sub"+i;
			Outfile_instances_arff(train_sub,trainfile,filepath);
			
		}
	}	
	
	//输出样本集信息
	public static void print_instances_info(Instances instances){
		int n0=0,n1=0;
		for(int i=0;i<instances.numInstances();i++ ){
			//System.out.println(s.instances.instance(i).stringValue(s.instances.numAttributes()-1).toString());
			if(instances.instance(i).stringValue(instances.numAttributes()-1).toString().equals("1"))
			{
				n1++;
				//System.out.println("样本"+i+"weight:"+instances.instance(i).weight());
			}
			else if(instances.instance(i).stringValue(instances.numAttributes()-1).toString().equals("0"))
			{
				n0++;		
				//System.out.println("样本"+i+"weight:"+instances.instance(i).weight());
			}
		}
		System.out.println("总样本数："+instances.numInstances()+"; 属性数:"+instances.numAttributes()+"; 类别数:"+instances.numClasses());
		System.out.println("0类样本数: "+n0+"; 1类样本数："+n1);
	}
	
	//输出二维double型矩阵信息
	public static void print_array(double A[][]){
		for(int i=0;i<A.length;i++){
			for(int j=0;j<A[i].length;j++){
				System.out.print(A[i][j]+"\t");
			}
			System.out.println();
		}
	}

	//得到分类器名称
	public static String Get_Classifier_name(Classifier classifier){
		String s=classifier.getClass().toString();  //s="class mulan.classifier.lazy.MLkNN"
		int Si=s.lastIndexOf(".")+1;
		int Ei=s.length();
		return s.substring(Si,Ei);  //classifierName="MLkNN" 分类器名称
	}
	
	//检查路径是否存在，若不存在则生成路径
	public static void Check_Path(String file_path){
        File file=new File(file_path);
        //判断目录，若不存在则新建
        if  (!file .exists()  && !file .isDirectory())      
        {       
            System.out.println("//不存在"+file_path+"需创建");  
            file.mkdir();    
        }
	}
	
	//返回数组的平均值
	public static double Get_Average(double A[]){
		double d=0.0;
		for(int i=0;i<A.length;i++){
			d+=A[i];
		}
		return d/A.length;
	}

	//返回一个在min和max之间的随机数（包括min和max）
	public static int random_int(int min,int max){
		Random r = new Random();
		int i=Math.abs(r.nextInt())%(max-min+1)+min;
		return i;
	}
	
	//返回一个在min和max之间的随机数（包括min和max）,presion为小数点后的精度
	public static double random_double(double min,double max, int presion){
		Random r = new Random();
		int dmin,dmax,t;
		t=1;
		while(presion>0){
			t*=10;
			presion--;
			
			//System.out.println(presion+"\t" +t);
		}
		
		dmin=(int)(t*min);
		dmax=(int)(t*max);
		
		
		int i=Math.abs(r.nextInt())%(dmax-dmin+1)+dmin;
		double d=(double)(i*1.0)/t;
		return d;
		
	}
	
	//数据集每类的样本数量
	public static int[] instances_class_num(Instances instances, String label[]){
		int label_num=label.length;
		int n[]=new int[label_num];
		for(int i=0;i<label_num;i++){
			n[i]=0;
		}
		int label_index=0;
		for(int i=0;i<instances.numInstances();i++ ){
			//System.out.println(s.instances.instance(i).stringValue(s.instances.numAttributes()-1).toString());
			for(int j=0;j<label.length;j++){
				if(instances.instance(i).stringValue(instances.numAttributes()-1).toString().equals(label[j]))
				{
					label_index=j;
					break;
				}
			}
			n[label_index]++;
		}
		return n;
	}
	
	public static String[] calLable(String result[][]){
		String[] res = new String[result[0].length];
		for(int i = 0;i <result[0].length;i++){
			int znum = 0;int onum = 0;
			for(int j = 0;j < result.length;j++){
				if(result[j][i].equals("0")){
					znum++;
				}else {
					onum++;
				}
				
			}
			if(znum>onum){
				res[i] = "0";
			}
			else{
				res[i] = "1";
			}
		}
		return res;
	}

	//输入每个子空间的分类结果result和每个子空间训练的基分类器的权重，输出集成后的结果
	public static float[][] caculate_ensemble_result(float predictResult[][][],double base_classfier_weight[]){
		int base_classifier_num=predictResult.length;
		int ins_num=predictResult[0].length;
		float ensembleResult[][]=new float[ins_num][1];
					
					for(int i = 0;i <ins_num;i++){
						ensembleResult[i][0]=0;
						for(int j = 0;j <base_classifier_num;j++){
							ensembleResult[i][0]+=predictResult[j][i][0]*base_classfier_weight[j];
						}
			
					}
					return ensembleResult;
				}
	
	//文件完全名称fileallname，out输出内容,Is_add为true时在文件末尾添加内容，为false时覆盖原先内容
	public static void Out_file(String fileallname,String out,boolean Is_add){
		try{
			FileWriter fout=new FileWriter(fileallname,Is_add);
			fout.write(out);
			fout.close();
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}
	
}
