import pdb
 
 
def sum(a, b):
    result = a * b
    return result

pdb.set_trace()
a = int(input("Enter a : "))
b = int(input("Enter b : "))
sum = sum(a, b)
print(f"Sum Result:", {sum})
