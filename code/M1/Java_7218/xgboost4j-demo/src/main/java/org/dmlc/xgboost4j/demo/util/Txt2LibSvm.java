package org.dmlc.xgboost4j.demo.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Txt2LibSvm {
	 static BufferedReader br = null; //read the file to bufferedreader  
	    static int classification = 0;   //classification number  
	    static FileWriter fw = null;     //put the result to file  
	    public static void main(String[] args) {  
	        // TODO Auto-generated method stub  
	        String sourceFileName = "C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\train_final_nouid.arff";   
	        String destFileName = "C:\\Users\\dell\\Documents\\GitHub\\xgboost\\demo\\data\\aaa.libsvm.txt";  
	        uci2Libsvm(sourceFileName, destFileName);  
	    }  
	      
	    public static void uci2Libsvm(String sourceFileName,String destFileName){  
	        String strline = null;  
	          
	        //whether the file is exists  
	        File file = new File(sourceFileName);  
	        if(!file.exists()){  
	            System.out.println("file not exists!");  
	            return;  
	        }  
	          
	        try {  
	            br = new BufferedReader(new FileReader(sourceFileName));  
	            fw = new FileWriter(destFileName);  
	               //the index of the libsvm format file   
//	            System.out.println("begin transform!");  
	            while((strline = br.readLine()) != null){  
	                String[] elements = strline.trim().split(",");  
	                  
	                if(elements.length < 10){  
	                    continue;  
	                } 
	                
	                classification = Integer.parseInt(elements[elements.length-1]);
	                String result=classification + " ";	                  

	                for(int i = 1;i<elements.length - 1;i++){
	                	result+= i + ":" + Float.parseFloat(elements[i-1]) ;
	                	result+= " " ;
	                }
	                result+= elements.length - 1 + ":" + Float.parseFloat(elements[elements.length - 2]);
	                
	               
//	                System.out.println(result);  
	                fw.write(result.trim() + "\n");   
	               
	            }  
	            fw.close();  
	            br.close();  
	        } catch (FileNotFoundException e) {  
	            // TODO Auto-generated catch block  
	            e.printStackTrace();  
	        } catch (IOException e) {  
	            // TODO Auto-generated catch block  
	            e.printStackTrace();  
	        }finally{  
	        }  
	          
//	        System.out.println("succeed!");  
	    }  

}


