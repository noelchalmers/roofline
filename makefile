#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#####################################################################################

#absolute path
ROOFLINE_DIR:=$(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

#include OCCA
OCCA_DIR=${ROOFLINE_DIR}/occa

#compilers to use for C/C++
CXX= g++

all: roofline

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

roofline: roofline.cpp ${OCCA_DIR}/lib/libocca.so
	$(CXX) -o roofline roofline.cpp -I${OCCA_DIR}/include -I${OCCA_DIR}/src -DROOFLINE_DIR='"${ROOFLINE_DIR}"' -Wl,-rpath,$(OCCA_DIR)/lib -L$(OCCA_DIR)/lib -locca

#cleanup
clean:
	rm -f roofline

realclean: clean
	${MAKE} -C ${OCCA_DIR} clean

