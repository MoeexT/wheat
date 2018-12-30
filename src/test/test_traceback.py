
import traceback

try:
    print("error coming...")
    print(2/0)
except Exception as e:
    print("catch error...")
    # print(traceback.print_exc(e))
    print(e.args)
    print("final...")
