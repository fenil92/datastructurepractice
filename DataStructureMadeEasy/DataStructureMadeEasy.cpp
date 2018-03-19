// DataStructureMadeEasy.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <queue>
using namespace std;
#pragma region TowerofHanoi


class TowerofHanoi
{
public:
	TowerofHanoi();
	~TowerofHanoi();
	void towers(int num, char frompeg, char topeg, char auxpeg)
	{
		if (num == 1)
		{
			cout << "n Move disk 1 from peg " << frompeg << " to peg " << topeg;
			return;
		}
		
		
		towers(num - 1, frompeg, auxpeg, topeg); //move n-1 peg from a to c using b
		cout << "n Move disk " << num << " from peg " << frompeg << " to peg " << topeg; //move nth peg from a to b
		towers(num - 1, auxpeg, topeg, frompeg); //move n-1 peg from c to b using a
	}
private:

};

TowerofHanoi::TowerofHanoi()
{
}

TowerofHanoi::~TowerofHanoi()
{
}
#pragma endregion

#pragma region LinkedList
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {} //constructor

};

class LinkedList
{
public:
	LinkedList();
	~LinkedList();
	void Traverse(ListNode *head)
	{
		int count = 0;
		ListNode *cur = head;
		while (cur != NULL)
		{
			cout << cur->val;
			cur = cur->next;
			count++;
		}
		cout << endl << "Length of LinkedList " << count << endl;
	}
	void InsertNode(ListNode **head, int data, int pos)
	{
		int count = 1;
		ListNode *newNode = new ListNode(data);
		ListNode *cur = *head;
		if (cur == NULL) //Linkedlist is empty
		{
			*head = newNode;
		}
		else
		{
			while (cur != NULL)
			{
				++count;
				if (pos == 1)
				{
					//head node insert
					newNode->next = cur;
					*head = newNode;
					break;
				}
				else if (count == pos)
				{
					//insert in middle or end
					newNode->next = cur->next;
					cur->next = newNode;
					break;
				}

				cur = cur->next;
			}
		}
	}

	void DeleteNode(ListNode **head, int data)
	{
		ListNode *cur = *head;
		int pos = 0;
		int count = 0;
		while (cur != NULL)
		{

			if (cur->val == data)
			{
				break;
			}
			cur = cur->next;
			pos++;
		}
		cur = *head;
		for (int i = 0;i <= pos;i++)
		{
			if (pos == 0)
			{
				*head = cur->next;
				break;
			}
			if (count + 1 == pos)
			{
				cur->next = cur->next->next;
			}
			cur = cur->next;
			count++;
		}
	}

	void RemoveNthFromEnd(ListNode* head, int n)
	{
		ListNode *cur = head;

		int pos = 0;
		int count = 0;
		while (cur != NULL)
		{
			cur = cur->next;
			pos++;
		}
		cur = head;

		for (int i = 1;i < pos - n;i++)
		{

			cur = cur->next;

		}
		if (n == 1)
		{
			cur->next = NULL;
		}
		else {
			cur->next = cur->next->next;
		}
	}
	//leetcode
	ListNode* removeNthFromEnd(ListNode* head, int n) {
		ListNode *cur = head;
		int pos = 0;

		while (cur != NULL)
		{
			cur = cur->next;
			pos++;
		}
		cur = head;

		for (int i = 0;i < pos - n - 1;i++)
		{

			cur = cur->next;

		}
		if (pos - n == 0) {
			head = head->next;
		}
		else if (cur != nullptr && n == 1)
		{
			cur->next = NULL;
		}
		else if (cur != nullptr && cur->next != nullptr) {
			cur->next = cur->next->next;
		}
		return head;
	}
	//leetcode
	ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {

		ListNode *l3 = NULL;
		ListNode *copyofl3;
		ListNode **mergedList = &copyofl3;
		if (l1 == nullptr)
		{
			l3 = l2;
			copyofl3 = l3;
			mergedList = &copyofl3;
		}
		if (l2 == nullptr)
		{
			l3 = l1;
			copyofl3 = l3;
			mergedList = &copyofl3;
		}
		while (l1 != NULL && l2 != NULL)
		{
			if (l1->val <= l2->val)
			{
				if (l3 == nullptr)
				{
					l3 = new ListNode(l1->val);
					copyofl3 = l3;
					mergedList = &copyofl3;
				}
				else {
					l3->next = new ListNode(l1->val);
					l3 = l3->next;
				}


				if (l1 != NULL) { l1 = l1->next; }
			}
			else
			{
				if (l3 == nullptr)
				{
					l3 = new ListNode(l2->val);
					copyofl3 = l3;
					mergedList = &copyofl3;
				}
				else {
					l3->next = new ListNode(l2->val);
					l3 = l3->next;
				}
				if (l2 != NULL) { l2 = l2->next; }
			}

		}
		if (l1 != nullptr) {
			while (l1 != nullptr)
			{
				l3->next = new ListNode(l1->val);
				l3 = l3->next;
				l1 = l1->next;
			}
		}

		if (l2 != nullptr) {
			while (l2 != nullptr)
			{
				l3->next = new ListNode(l2->val);
				l3 = l3->next;
				l2 = l2->next;
			}
		}
		return *mergedList;

	}
	//leetcode
	ListNode* mergeKLists(vector<ListNode*>& lists) {

	}
	//Leetcode - Given a sorted linked list, delete all duplicates such that each element appear only once.
	ListNode* deleteDuplicates(ListNode* head) {
		ListNode *cur = head;
		ListNode *prev;
		vector<int> v;
		if (cur == nullptr) { return head; }

		while (cur != nullptr)
		{
			if (find(v.begin(), v.end(), cur->val) != v.end())
			{
				prev->next = cur->next;
			}
			else
			{
				v.push_back(cur->val);
				prev = cur;
			}
			cur = cur->next;
		}


		return head;
	}

	ListNode* rotateRight(ListNode* head, int k) {
		if (head == nullptr) { return head; }

		ListNode* cur = head;
		ListNode* end;


		int length = 1;
		//calculate length of list
		while (cur->next != nullptr)
		{
			length++;
			cur = cur->next;
		}
		//if length is 1 or k equals length then don't rotate
		if (length == 1 || k % length == 0) {
			return head;
		}

		end = cur;
		cur->next = head;
		int steps = length - k % length;
		for (int i = 0;i < steps;i++)
		{
			cur = cur->next;
		}
		//setting new head node
		end = cur->next;
		cur->next = nullptr;

		return end;
	}

	ListNode* oddEvenList(ListNode* head) {
		if (head == nullptr || head->next == nullptr) { return head; }
		ListNode *odd = head;
		ListNode *even = head->next;
		ListNode *startEven = head->next;
		while (even->next != nullptr && even->next->next != nullptr)
		{

			odd->next = even->next;
			odd = odd->next;
			even->next = odd->next;
			even = even->next;
		}

		odd->next = startEven;
		return head;
	}

	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {

		if (headA == nullptr || headB == nullptr) { return nullptr; }
		int lenA = lengthofList(headA);
		int lenB = lengthofList(headB);
		int diff = abs(lenA - lenB);
		if (lenA > lenB)
		{
			for (int i = 0; i < diff;i++)
			{
				headA = headA->next;
			}
			while (headA != headB)
			{
				headA = headA->next;
				headB = headB->next;
			}
			return headA;
		}
		else
		{
			for (int i = 0; i < diff;i++)
			{
				headB = headB->next;
			}
			while (headA != headB)
			{
				headA = headA->next;
				headB = headB->next;
			}
			return headA;
		}

	}
	int lengthofList(ListNode * head)
	{
		int len = 0;
		while (head->next != nullptr)
		{
			head = head->next;
			len++;
		}
		return len;
	}

	ListNode *detectCycle(ListNode *head) {
		if (head == nullptr || head->next == nullptr) { return nullptr; }
		ListNode *fast = head;
		ListNode *slow = head;

		while (fast->next != nullptr && fast->next->next != nullptr)
		{

			slow = slow->next;
			fast = fast->next->next;

			if (slow == fast)
			{
				break;
			}
		}

		//No cycle
		if (fast->next == nullptr || fast->next == nullptr) { return nullptr; }

		slow = head;
		while (slow != fast)
		{
			slow = slow->next;
			fast = fast->next;
		}

		return slow;
	}

	void reorderList(ListNode* head) {
		if (head != nullptr && head->next != nullptr)
		{

			ListNode *cur = head;
			vector<int> v;
			int len = 0;
			while (cur != nullptr)
			{
				v.push_back(cur->val);
				cur = cur->next;
				len++;
			}
			cur = head;
			int i = 0;
			for (i = 0; i < len / 2;i++)
			{
				if (i != 0)
				{
					cur->next = new ListNode(v[i]);
					cur = cur->next;
				}

				cur->next = new ListNode(v[len - i - 1]);
				cur = cur->next;
			}

			if (len % 2 != 0)
			{
				cur->next = new ListNode(v[i]);
			}

		}
	}

	ListNode* reverseBetween(ListNode* head, int m, int n) {

		if (head == nullptr || head->next == nullptr) { return head; }
		ListNode *prev = nullptr;
		ListNode *cur = head;
		ListNode *end = nullptr;
		int i = 1;

		while (i < n)
		{
			if (i + 1 == m || m == 1)
			{
				prev = cur;
			}


			if (i == m)
			{

				while (i <= n)
				{
					ListNode *next = cur->next;
					cur->next = end;
					end = cur;
					cur = next;
					i++;
				}
				if (m == 1) {

					prev = end;
					while (prev->next != nullptr)
					{
						prev = prev->next;
					}
					prev->next = cur;
					return end;
				}
				else {
					prev->next = end;
					while (prev->next != nullptr)
					{
						prev = prev->next;
					}
					prev->next = cur;
					goto label;

				}



			}
			cur = cur->next;
			i++;
		}
	label:return head;

	}


	ListNode* InsertionsortList(ListNode* head) {
		if (head == nullptr || head->next == nullptr)
			return head;

		ListNode *newHead = new ListNode(head->val);
		ListNode *pointer = head->next;

		// loop through each element in the list
		while (pointer != nullptr) {
			// insert this element to the new list

			ListNode *innerPointer = newHead;
			ListNode *next = pointer->next;

			if (pointer->val <= newHead->val) {
				ListNode *oldHead = newHead;
				newHead = pointer;
				newHead->next = oldHead;
			}
			else {
				while (innerPointer->next != nullptr) {

					if (pointer->val > innerPointer->val && pointer->val <= innerPointer->next->val) {
						ListNode *oldNext = innerPointer->next;
						innerPointer->next = pointer;
						pointer->next = oldNext;
					}

					innerPointer = innerPointer->next;
				}

				if (innerPointer->next == nullptr && pointer->val > innerPointer->val) {
					innerPointer->next = pointer;
					pointer->next = nullptr;
				}
			}

			// finally
			pointer = next;
		}

		return newHead;
	}

private:

};


LinkedList::LinkedList()
{
}

LinkedList::~LinkedList()
{
}
#pragma endregion

#pragma region DoublyLinkedList
struct DoublyListNode {
	int val;
	DoublyListNode *next;
	DoublyListNode *previous;
	DoublyListNode(int x) : val(x), next(NULL),previous(NULL) {} //constructor

};

class DoublyLinkedList
{
public:
	DoublyLinkedList();
	~DoublyLinkedList();
	void InsertNode(DoublyListNode **head,int data)
	{
		int count = 0;
		DoublyListNode *cur = *head;
		if (cur == NULL)
		{
			cur = new DoublyListNode(data);
		}
		else
		{
			while (cur != NULL)
			{
				cur = cur->next;
				if (cur->next == NULL)
				{
					cur->next = new DoublyListNode(data);
					cur->next->previous = cur;
					break;
				}
			}
		}
	}
	void DeleteNode(DoublyListNode **head, int data)
	{
		DoublyListNode *cur = *head;
		
			while (cur != NULL)
			{
				if (cur->val == data)
				{
					if (cur->previous == NULL)
					{
						//first node
						*head=cur->next;
						cur->next = NULL;
						break;
					}
					else if (cur->next==NULL)
					{
						cur->previous->next = NULL;
					}
					else {
						cur->next->previous = cur->previous;
						cur->previous->next = cur->next;
						break;
					}
					
				}
				cur = cur->next;
			}
		}
	
private:

};

DoublyLinkedList::DoublyLinkedList()
{
}

DoublyLinkedList::~DoublyLinkedList()
{
}
#pragma endregion

int main()
{
//#pragma region TowerofHanoiCall
//	int num;
//	cout << "Enter the number of disks : ";
//	cin >> num;
//	cout << "The sequence of moves involved in the Tower of Hanoi are :n";
//	TowerofHanoi t;
//	t.towers(num, 'A', 'B', 'C');
//	return 0;

//#pragma endregion
//#pragma region LinkedListCall
//	LinkedList l;
//	ListNode *l1 = new ListNode(3);
//	l1->next = new ListNode(5);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(7);
//	ListNode **head = &l2;
//	l.Traverse(*head);
//	l.InsertNode(head, 9, 4);
//	l.Traverse(*head);
//	//l.DeleteNode(head,3);
//	l.RemoveNthFromEnd(*head, 2);
//	return 0;
//#pragma endregion	
 
//#pragma region DoublyLinkedListCall
//	DoublyLinkedList dl;
//	DoublyListNode *dl1 = new DoublyListNode(3);
//	auto dl2 = dl1;
//	auto **head = &dl2;
//	dl1->previous = NULL;
//	dl1->next = new DoublyListNode(5);
//	dl1->next->previous = dl1;
//	dl1 = dl1->next;
//	//dl1->previous = *dl2;
//	dl1->next = new DoublyListNode(7);
//	dl1->next->previous = dl1;
//	dl.InsertNode(head,9);
//	dl.DeleteNode(head,9);
//#pragma endregion

//#pragma region MergeSortedLinkedList
//	LinkedList l;
//	ListNode *l1 = new ListNode(3);
//	l1->next = new ListNode(6);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(7);
//	ListNode **head = &l2;
//	l.Traverse(*head);
//	ListNode *l4 = new ListNode(2);
//	l4->next = new ListNode(4);
//	auto l3 = l4;
//	ListNode **head2 = &l3;
//	l4 = l4->next;
//	l4->next = new ListNode(5);
//	l.mergeTwoLists(*head,*head2);
//#pragma endregion

//#pragma region mergeKLists
//	LinkedList l;
//	ListNode *l1 = new ListNode(3);
//	l1->next = new ListNode(6);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(7);
//	ListNode **head = &l2;
//	l.Traverse(*head);
//	ListNode *l4 = new ListNode(2);
//	l4->next = new ListNode(4);
//	auto l3 = l4;
//	ListNode **head2 = &l3;
//	l4 = l4->next;
//	l4->next = new ListNode(5);
//	vector<ListNode*> List;
//	List.push_back(*head);
//	List.push_back(*head2);
//
//#pragma endregion


//#pragma region rotateRight
//	LinkedList l;
//	ListNode *l1 = new ListNode(1);
//	l1->next = new ListNode(2);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(3);
//	ListNode **head = &l2;
//	l.rotateRight(*head, 2);
//#pragma endregion

//#pragma region oddEvenList
//	LinkedList l;
//	ListNode *l1 = new ListNode(1);
//	l1->next = new ListNode(2);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(3);
//	//l1->next->next = new ListNode(4);
//	ListNode **head = &l2;
//	l.oddEvenList(*head);
//#pragma endregion

//#pragma region detectCycle
//	LinkedList l;
//	ListNode *l1 = new ListNode(1);
//	l1->next = new ListNode(2);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(3);
//	l1->next->next = new ListNode(4);
//	l1->next->next->next = new ListNode(5);
//	l1->next->next->next->next = l1->next;
//	ListNode **head = &l2;
//	ListNode *beginCycle= l.detectCycle(*head);
//#pragma endregion

//#pragma region reorderList
//	LinkedList l;
//	ListNode *l1 = new ListNode(1);
//	l1->next = new ListNode(2);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(3);
//	//l1->next->next = new ListNode(4);
//	ListNode **head = &l2;
//	l.reorderList(*head);
//#pragma endregion

//#pragma region reverseBetween
//	LinkedList l;
//	ListNode *l1 = new ListNode(1);
//	l1->next = new ListNode(2);
//	auto l2 = l1;
//	l1 = l1->next;
//	l1->next = new ListNode(3);
//	l1->next->next = new ListNode(4);
//	l1->next->next->next = new ListNode(5);
//	ListNode **head = &l2;
//	l.reverseBetween(*head, 2,5);
//#pragma endregion

#pragma region sortList
	LinkedList l;
	ListNode *l1 = new ListNode(5);
	l1->next = new ListNode(4);
	auto l2 = l1;
	l1 = l1->next;
	l1->next = new ListNode(3);
	/*l1->next->next = new ListNode(4);
	l1->next->next->next = new ListNode(2);*/
	ListNode **head = &l2;
	l.InsertionsortedList(*head);
#pragma endregion
}

