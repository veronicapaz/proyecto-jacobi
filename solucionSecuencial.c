#include <stdio.h>
#include <stdlib.h>
#define true 1
#define n 100

int i =50;
int j =50;
int new[i][j];
int grid[i][j];
double maxdiff = 0.0;
double max =0.0;

int main(int argc, char  **argv){
	while(true){
		//compute new values for all interior points
		for (i = 1; i <=n; ++i)
		{
			for (j = 1; j < n; ++j)
			{
				new[i,j] = (grid[i-1,j]+grid[i+1,j]+grid[i,j-1]+grid[i,j+1])/4;
				//inters++; q nose q es
			}
		}

		//compute the maximum difference
		for (i = 1; i <=n; ++i)
		{
			for (j = 1; j < n; ++j)
			{
				maxdiff = max(maxdiff, abs(new[i,j]-grid[i,j]));
			}
			//check for termination
			if(maxdiff<EPSILON)
				break;
			//copy new to grid to prepare for a next updates
			for (i = 1; i <=n; ++i)
			{
				for (j = 1; j < n; ++j)
				{
					grid[i,j] = new[i,j];
				}
		}
	}
}
}
