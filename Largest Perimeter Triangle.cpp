class Solution {
public:
    int largestPerimeter(vector<int>& nums) {
       int n = nums.size(); 
       sort(nums.begin(), nums.end());

        int ans = 0;
        int i = 1;
        while(i < n-1){
            if(nums[i-1] + nums[i] > nums[i+1]){
                ans = max(ans,nums[i-1] + nums[i] + nums[i+1]);
            }
            i++;
        }
        return ans;

    }
};
