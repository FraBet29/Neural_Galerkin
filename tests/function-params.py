def first_function(x, y):

	def second_function(x):
		return x + y
	
	for _ in range(10):
		res = second_function(x)
		print(res)
		y += 1


first_function(1, 2)