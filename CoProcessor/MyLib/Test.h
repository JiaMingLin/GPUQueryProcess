#ifndef TEST_H
#define TEST_H

void testPrimitive(int argc, char **argv);
void testJoin(int argc, char **argv);

//test the single primitive
void testReduce( int rLen, int numThread, int OPERATOR );
void testFilter( int rLen, int numThread );

#endif
