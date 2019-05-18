#!/usr/bin/env python
# coding: utf-8

# In[149]:


from pyspark import SparkContext,SparkConf
import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructField,StringType,DoubleType,IntegerType
from pyspark.sql import functions as f
from pyspark.sql.functions import lit,trim,concat,coalesce,udf,struct
from pyspark.sql import SQLContext


# In[4]:


sc = SparkContext('local','similarity')
sqlContext = SQLContext(sc)


# In[150]:


#reading the input
input_1 = sc.textFile('./data1.txt')


# In[151]:


input_f=input_1.map(lambda x: x+'~ ') #delimiting every document by ~


# In[152]:


input_flat = input_f.flatMap(lambda x: x.split(' '))


input_dist = input_flat.distinct()


input_filter = input_dist.filter(lambda x: x is not None and x!="")


# In[153]:


count_non_zero=udf(lambda row:len([x for x in row if x!=None]),IntegerType()) #udf to get count of non_zero columns
sqr=udf(lambda row:(sum([x**2 for x in row if x!=None]))**(1/2),DoubleType()) #udf to get sqrt_of_sum 


# In[178]:


#calculating tf

data=input_flat.collect()

dffilt = sqlContext.createDataFrame(input_filter.collect(), StringType())
for i in range(input_1.count()): 
    d1=data.index('~')
    #print(i,d1)
    df = sqlContext.createDataFrame(data[0:d1], StringType())
    data=data[d1+1:]
    df_1=df.withColumn('d'+str(i),f.lit(1)) #adding col for each documnet
    #calculating tf 
    df_f=df_1.groupBy('value').agg(f.round((f.count('d'+str(i))/d1),3).alias('d'+str(i))) #rounding off data to 3 decimal places
    dffilt=dffilt.join(df_f,"value",'left') #adding each col to main dataframe
    


# In[157]:


dffilt.show()


# In[179]:


dffilt=dffilt.withColumn('count',count_non_zero(struct([dffilt[x] for x in dffilt.columns]))-1) #counting non_zero columns and storing in count


# In[180]:


dffilt=dffilt.fillna(0) #replace null with 0


# In[181]:


#calculating IDF
for i in range(input_1.count()): 
    dffilt=dffilt.withColumn('d'+str(i),f.round(f.col('d'+str(i))*f.log(input_1.count()/f.col('count')),3)) #rounding off data to 3 decimal places


# In[182]:


#dffilt.show()


# In[183]:


dffilt=dffilt.where(f.col('value') != '~') #removing ~


# In[184]:


#Testing data for two words "few" and "input"

dffilt_test=dffilt.where((f.col('value') == 'few') | (f.col('value') == 'input'))


# In[185]:


#dffilt_test.show()


# In[ ]:


cols=dffilt_test.columns
cols.remove('value')
cols.remove('count')
dffilt_test=dffilt_test.drop('value')
dffilt_test=dffilt_test.drop('count')
#Removed not needed columns
dffilt_test2=dffilt_test.withColumn('sqr',sqr(struct([dffilt_test[x] for x in cols]))) #getting denominator


# In[ ]:


#dffilt_test2.show()


# In[ ]:


transposed_df = sqlContext.createDataFrame(dffilt_test2.toPandas().transpose().reset_index()) #getting transpose


# In[ ]:


#transposed_df.show()


# In[ ]:


transposed_df=transposed_df.withColumn("mult",f.col('0')*f.col('1')) #getting final values of numerator and denominator(sqr)


# In[ ]:


#transposed_df.show()


# In[ ]:


#dividing numerator and denominator
answer=transposed_df.where(f.col('index') != 'sqr').agg(f.sum('mult')).rdd.collect()[0][0]/transposed_df.select('mult').where(f.col('index') == 'sqr').rdd.collect()[0][0]


# In[ ]:


print(answer)

