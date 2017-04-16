--- ld_model.py	(original)
+++ ld_model.py	(refactored)
@@ -51,7 +51,7 @@
 	for num in range(0,len(Y_result)):
 		for s in range(0,len(Y_result[num])):
 			Y_result[num][s] =int(round(Y_result[num][s]))
-	out=map(int,Y_result[0].tolist())
+	out=list(map(int,Y_result[0].tolist()))
 	return out
 
 def Alstm(b):
@@ -67,7 +67,7 @@
 	for num in range(0,len(Y_result)):
 		for s in range(0,len(Y_result[num])):
 			Y_result[num][s] =int(round(Y_result[num][s]))
-	out=map(int,Y_result[0].tolist())
+	out=list(map(int,Y_result[0].tolist()))
 	return out
 
 def Plstm(b):
@@ -86,13 +86,13 @@
 	for num in range(0,len(Y_result)):
 		for s in range(0,len(Y_result[num])):
 			Y_result[num][s] =int(round(Y_result[num][s]))
-	out=map(int,Y_result[0].tolist())
+	out=list(map(int,Y_result[0].tolist()))
 	return out
 
 a=Rlstm([1,2,3,1])
-print a
+print(a)
 a=Rlstm([2,4,5,1])
-print a
+print(a)
 a=Rlstm([3,6,6,6])
-print a
+print(a)
 
