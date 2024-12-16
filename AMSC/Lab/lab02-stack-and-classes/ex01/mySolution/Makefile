# Mikefile template

# directly
cpp = main.cpp
obj = 
exe = main.exe

# multiple reference
# cpp = ${shell find -name *.cpp}
# obj = ${subst .cpp,.o,${cpp}}
# exe = main.exe

.PHONY : debug compile exec

#execution
exec : ${cpp}
	g++ ${cpp} -o ${exe}

# compile
${obj} : ${cpp}
	g++ -c ${cpp} -o ${obj}

compile : ${obj}

#debug
debug : 
	@echo ${cpp}



