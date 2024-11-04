CC=g++
CXXFLAGS=-Wall -pedantic -std=c++11 -O3
LDFLAGS=-L. -lcec2013lsgo


OBJECTS=Benchmarks.o \
F1.o F2.o F3.o F4.o F5.o F6.o F7.o F8.o F9.o F10.o\
F11.o F12.o F13.o F14.o F15.o eval_func.o

default: libcec2013lsgo.so demo demo2

# Regla para crear la biblioteca compartida
libcec2013lsgo.so: $(OBJECTS)
	$(CC) -shared -o libcec2013lsgo.dll $(OBJECTS)

# Regla para crear el ejecutable demo y enlazarlo con la biblioteca compartida
demo: demo.o libcec2013lsgo.dll
	$(CC) $(CXXFLAGS) -o demo demo.o $(LDFLAGS)

# Regla para crear el ejecutable demo2 y enlazarlo con la biblioteca compartida
demo2: demo2.o libcec2013lsgo.dll
	$(CC) $(CXXFLAGS) -o demo2 demo2.o $(LDFLAGS)

demo.o: demo.cpp Header.h Benchmarks.h
	$(CC) $(CXXFLAGS) -c demo.cpp

demo2.o: demo2.cpp
	$(CC) $(CXXFLAGS) -c demo2.cpp

Benchmarks.o: Benchmarks.h Benchmarks.cpp
	$(CC) $(CXXFLAGS) -c Benchmarks.cpp

F1.o: F1.h Benchmarks.h F1.cpp
	$(CC) $(CXXFLAGS) -c F1.cpp

F2.o: F2.h Benchmarks.h F2.cpp
	$(CC) $(CXXFLAGS) -c F2.cpp

F3.o: F3.h Benchmarks.h F3.cpp
	$(CC) $(CXXFLAGS) -c F3.cpp

F4.o: F4.h Benchmarks.h F4.cpp
	$(CC) $(CXXFLAGS) -c F4.cpp

F5.o: F5.h Benchmarks.h F5.cpp
	$(CC) $(CXXFLAGS) -c F5.cpp

F6.o: F6.h Benchmarks.h F6.cpp
	$(CC) $(CXXFLAGS) -c F6.cpp

F7.o: F7.h Benchmarks.h F7.cpp
	$(CC) $(CXXFLAGS) -c F7.cpp

F8.o: F8.h Benchmarks.h F8.cpp
	$(CC) $(CXXFLAGS) -c F8.cpp

F9.o: F9.h Benchmarks.h F9.cpp
	$(CC) $(CXXFLAGS) -c F9.cpp

F10.o: F10.h Benchmarks.h F10.cpp
	$(CC) $(CXXFLAGS) -c F10.cpp

F11.o: F11.h Benchmarks.h F11.cpp
	$(CC) $(CXXFLAGS) -c F11.cpp

F12.o: F12.h Benchmarks.h F12.cpp
	$(CC) $(CXXFLAGS) -c F12.cpp

F13.o: F13.h Benchmarks.h F13.cpp
	$(CC) $(CXXFLAGS) -c F13.cpp

F14.o: F14.h Benchmarks.h F14.cpp
	$(CC) $(CXXFLAGS) -c F14.cpp

F15.o: F15.h Benchmarks.h F15.cpp
	$(CC) $(CXXFLAGS) -c F15.cpp

clean:
	rm -f demo demo2 libcec2013lsgo.dll $(OBJECTS) demo.o demo2.o
