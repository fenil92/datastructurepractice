```c#
using System;
using System.Linq;

class Program
{
    static void Main()
    {
        int N = 3;
        int M = 5;
        int[,] marks = {
            { 75, 76, 65, 87, 87 },
            { 78, 76, 68, 56, 89 },
            { 67, 87, 78, 77, 65 }
        };

        int[] totalMarks = CalculateTotalMarks(N, M, marks);

        Console.WriteLine(string.Join(", ", totalMarks));
    }

    static int[] CalculateTotalMarks(int N, int M, int[,] marks)
    {
        // Step 1: Calculate the class average for each subject
        double[] classAverages = new double[M];
        for (int j = 0; j < M; j++)
        {
            double sum = 0;
            for (int i = 0; i < N; i++)
            {
                sum += marks[i, j];
            }
            classAverages[j] = sum / N;
        }

        // Step 2: Identify the subject with the lowest class average
        int subjectToIgnore = Array.IndexOf(classAverages, classAverages.Min());

        // Step 3: Calculate the total marks for each student excluding the ignored subject
        int[] totalMarks = new int[N];
        for (int i = 0; i < N; i++)
        {
            int sum = 0;
            for (int j = 0; j < M; j++)
            {
                if (j != subjectToIgnore)
                {
                    sum += marks[i, j];
                }
            }
            totalMarks[i] = sum;
        }

        return totalMarks;
    }
}
```
