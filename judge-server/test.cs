using System;

namespace APlusB
{
    class Program
    {
        static void Main(string[] args)
        {
            var input = Console.ReadLine();
            var tokens = input.Split(' ');
            var a = long.Parse(tokens[0]);
            var b = long.Parse(tokens[1]);
            Console.WriteLine(a + b);
        }
    }
}
