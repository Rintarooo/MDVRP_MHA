//
//  types.h
//  MDVRP
//
//  Created by Mathias Aarseth Pedersen on 06/02/2018.
//  Copyright © 2018 Mathias Aarseth Pedersen. All rights reserved.
//

#ifndef types_h
#define types_h
#include <cmath>

template <typename T>
using upVec = std::unique_ptr<std::vector<T>>;
template <typename T>
using spVec = std::shared_ptr<std::vector<T>>;




#endif /* types_h */
