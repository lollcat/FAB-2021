# Notes
## Training
 - when we move to a more complicate example with flows training instability becomes more of an issue
   - ceiling log p_x
    - big batch size
    - could initialise variance higher
 - it needs to find high density regions to latch on to. 
   If it can't find these then the MC estimate does the thing where it makes q(x) high and far away from p(x), so the
   punishment zones with high p(x) and low q(x) aren't sampled from