# Mikefile template

# directly
cpp = memoryTest.cpp
obj = 
exe = main.exe

# multiple reference
# cpp = ${shell find -name *.cpp}
# obj = ${subst .cpp,.o,${cpp}}
# exe = main.exe

.PHONY : debug compile exec

#execution
exec : ${cpp} ${exe}
	g++ ${cpp} -o ${exe}

# compile
${obj} : ${cpp}
	g++ -c ${cpp} -o ${obj}

compile : ${obj}

#debug
debug : 
	@echo ${cpp}



