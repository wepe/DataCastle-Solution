package org.dmlc.xgboost4j.demo.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Txt2Arff {
	
   public static void txt2Arff(String inputfile,String outputfile) throws IOException{
	   FileReader FR = new FileReader(inputfile);
	   BufferedReader BR = new BufferedReader(FR);
	   
	   FileWriter FW = new FileWriter(outputfile);
	   BufferedWriter BW = new BufferedWriter(FW);
	   
	   String line = BR.readLine();
	   String [] features = line.split(",");
	   int featureNum = features.length -1; 
	   
	   String arffHeader = "@relation CrossValiData-weka.filters.unsupervised.attribute.Normalize-S1.0-T0.0\n\n";
	   for(int i = 0;i<features.length-1;i++){
		   
		   arffHeader+="@attribute attr"+(i+1)+" numeric\n"; 
		   
	   }
	    arffHeader+="@attribute class {0,1}\n\n"; 	    
	    arffHeader+="@data"; 
	   BW.write(arffHeader);
	   BW.newLine();

	   while(line!= null){		   
		   BW.write(line);
		   BW.newLine();
		   line = BR.readLine(); 
	   }
	   
	   BR.close();
	   FR.close();
	   BW.close();
	   FW.close();
   }
	
   public static void main(String[] args) throws IOException {
	   String inputFile = "C:\\Users\\dell\\Desktop\\电子科大比赛\\20160121\\70_train";
	   txt2Arff(inputFile+".csv",inputFile+".arff ");
  }
}

