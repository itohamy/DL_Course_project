#include <stdio.h>
#include <fcntl.h>
void main()
{
	int fd=open("afile",O_WRONLY);
	printf("Test");
}


